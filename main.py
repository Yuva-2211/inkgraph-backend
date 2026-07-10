"""
InkGraph backend — FastAPI entrypoint.

Endpoints:
    GET  /health
    GET  /documents?user_id=<uuid>
    POST /documents
    GET  /documents/{id}
    GET  /documents/{id}/revisions
    POST /documents/{id}/decision
    GET  /documents/{id}/export/pdf

Run (dev):
    uvicorn main:app --reload

Run (prod):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from io import BytesIO
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth import get_current_user_id
from config import settings
from db import get_supabase
from graph.workflow import build_workflow

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
workflow_app = None

# Maps document_id (str) → LangGraph thread config dict.
# In production with multiple workers, move this to Redis.
_workflow_threads: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow_app
    workflow_app = build_workflow()
    yield
    workflow_app = None


app = FastAPI(title="InkGraph API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class CreateDocumentRequest(BaseModel):
    title: str
    prompt: str
    word_limit: int | None = None
    writing_style: str = "general"


class HumanDecisionRequest(BaseModel):
    decision: str       # "approved" | "changes"
    note: str | None = None


# ---------------------------------------------------------------------------
# Supabase helpers (sync — called inside thread pool)
# ---------------------------------------------------------------------------
def _update_doc(document_id: str, **fields) -> None:
    get_supabase().table("documents").update(fields).eq("id", document_id).execute()


def _add_revision(
    document_id: str,
    stage: str,
    content: str | None = None,
    note: str | None = None,
) -> None:
    get_supabase().table("revisions").insert(
        {"document_id": document_id, "stage": stage, "content": content, "note": note}
    ).execute()


# ---------------------------------------------------------------------------
# LangGraph background runners
# ---------------------------------------------------------------------------
def _handle_node_event(document_id: str, node_name: str, update: dict) -> None:
    """Translate a LangGraph node event into Supabase updates."""
    if node_name == "planner":
        outline = update.get("outline")
        _update_doc(document_id, status="writing", outline=outline)
        _add_revision(document_id, "planner",
                      content=json.dumps(outline),
                      note="Outline created by Planner Agent.")

    elif node_name == "search":
        search_results = update.get("search_results") or ""
        _add_revision(document_id, "system", note="Information fetched from internet.", content=search_results)

    elif node_name == "writer":
        draft = update.get("draft")
        cycle = update.get("review_cycle", 0)
        _update_doc(document_id, status="reviewing", current_content=draft)
        note = (
            f"Draft revised (cycle {cycle}) per reviewer feedback."
            if cycle > 0
            else "Initial draft written by Writer Agent."
        )
        _add_revision(document_id, "writer", content=draft, note=note)

    elif node_name == "fact_checker":
        notes = update.get("review_notes") or []
        latest = notes[-1] if notes else None
        if latest and "[Fact Checker" in latest:
            _add_revision(document_id, "reviewer", note=latest)

    elif node_name == "reviewer":
        notes = update.get("review_notes") or []
        latest = notes[-1] if notes else "Review complete."
        _add_revision(document_id, "reviewer", note=latest)

    elif node_name == "tone_optimizer":
        draft = update.get("draft")
        _update_doc(document_id, current_content=draft)
        _add_revision(
            document_id, "system",
            content=draft,
            note="Tone & style optimized. Awaiting human review.",
        )


def _run_workflow_sync(document_id: str, prompt: str, word_limit: int | None, writing_style: str) -> None:
    """
    Run the LangGraph graph synchronously (called via asyncio.to_thread).
    Streams events until the graph pauses at the human interrupt.
    """
    config = {"configurable": {"thread_id": document_id}}
    _workflow_threads[document_id] = config

    initial_state = {
        "document_id": document_id,
        "prompt": prompt,
        "outline": None,
        "draft": None,
        "review_notes": [],
        "needs_revision": False,
        "review_cycle": 0,
        "human_decision": None,
        "word_limit": word_limit,
        "writing_style": writing_style,
        "search_results": None,
    }

    try:
        _update_doc(document_id, status="planning")
        for event in workflow_app.stream(initial_state, config=config):
            for node_name, update in event.items():
                if node_name.startswith("__"):
                    continue
                _handle_node_event(document_id, node_name, update)
        # Stream ended — graph paused before human node
        _update_doc(document_id, status="awaiting_human")
    except Exception as exc:
        _update_doc(document_id, status="archived")
        _add_revision(document_id, "system", note=f"Error during workflow: {exc}")
        raise


def _resume_workflow_sync(document_id: str, decision: str, note: str | None = None) -> None:
    """
    Resume a paused graph (called via asyncio.to_thread).
    Injects human_decision into the checkpointed state and continues streaming.
    """
    config = _workflow_threads.get(document_id)
    if not config:
        # Reconstruct thread config on the fly (e.g. if backend reloaded/restarted)
        config = {"configurable": {"thread_id": document_id}}
        _workflow_threads[document_id] = config

    try:
        state_updates = {"human_decision": decision}
        if note:
            try:
                state_info = workflow_app.get_state(config)
                current_notes = state_info.values.get("review_notes") or []
                state_updates["review_notes"] = list(current_notes) + [f"[Human Feedback] {note}"]
            except Exception:
                pass

        # Ensure prompt and workflow parameters are preserved on resume
        state_info = workflow_app.get_state(config)
        if state_info and "prompt" in state_info.values:
            state_updates["prompt"] = state_info.values["prompt"]
            state_updates["word_limit"] = state_info.values.get("word_limit")
            state_updates["writing_style"] = state_info.values.get("writing_style")
        else:
            db_result = get_supabase().table("documents").select("prompt,word_limit,writing_style").eq("id", document_id).single().execute()
            if db_result.data:
                state_updates["prompt"] = state_updates.get("prompt") or db_result.data.get("prompt")
                state_updates["word_limit"] = state_updates.get("word_limit") or db_result.data.get("word_limit")
                state_updates["writing_style"] = state_updates.get("writing_style") or db_result.data.get("writing_style")

        workflow_app.update_state(config, state_updates)

        for event in workflow_app.stream(None, config=config):
            for node_name, update in event.items():
                if node_name.startswith("__"):
                    continue
                _handle_node_event(document_id, node_name, update)

        if decision == "approved":
            _update_doc(document_id, status="approved")
        else:
            _update_doc(document_id, status="awaiting_human")

    except Exception as exc:
        _add_revision(document_id, "system", note=f"Resume error: {exc}")
        raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    from version import get_version_info; return get_version_info()


@app.get("/documents")
async def list_documents(user_id: str = Depends(get_current_user_id)):
    result = (
        get_supabase()
        .table("documents")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


@app.post("/documents", status_code=201)
async def create_document(
    payload: CreateDocumentRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    """
    Inserts the document row and immediately kicks off the LangGraph pipeline
    as a FastAPI background task (non-blocking HTTP response).
    """
    result = (
        get_supabase()
        .table("documents")
        .insert({
            "user_id": user_id,
            "title": payload.title,
            "prompt": payload.prompt,
            "word_limit": payload.word_limit,
            "writing_style": payload.writing_style,
            "status": "planning",
        })
        .execute()
    )
    document = result.data[0]
    doc_id = document["id"]

    async def _bg():
        await asyncio.to_thread(_run_workflow_sync, doc_id, payload.prompt, payload.word_limit, payload.writing_style)

    background_tasks.add_task(_bg)
    return document


@app.get("/documents/{document_id}")
async def get_document(
    document_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    result = (
        get_supabase()
        .table("documents")
        .select("*")
        .eq("id", str(document_id))
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")
    return result.data[0]


@app.get("/documents/{document_id}/revisions")
async def get_revisions(
    document_id: UUID,
    _user_id: str = Depends(get_current_user_id),
):
    result = (
        get_supabase()
        .table("revisions")
        .select("*")
        .eq("document_id", str(document_id))
        .order("created_at", desc=False)
        .execute()
    )
    return result.data


@app.post("/documents/{document_id}/decision")
async def submit_decision(
    document_id: UUID,
    payload: HumanDecisionRequest,
    background_tasks: BackgroundTasks,
    _user_id: str = Depends(get_current_user_id),
):
    """
    Human-in-the-loop gate.
    - 'approved' → marks document final, graph ends.
    - 'changes'  → resumes graph from writer node with feedback.
    """
    if payload.decision not in ("approved", "changes"):
        raise HTTPException(
            status_code=400, detail="decision must be 'approved' or 'changes'"
        )

    doc_id = str(document_id)

    # Immediate status update
    next_status = "approved" if payload.decision == "approved" else "revising"
    get_supabase().table("documents").update({"status": next_status}).eq("id", doc_id).execute()

    # Log the human decision
    note = payload.note or (
        "Approved as final." if payload.decision == "approved" else "Changes requested — routing back to Writer."
    )
    _add_revision(doc_id, "human", note=note)

    # Resume in background
    decision = payload.decision
    feedback_note = payload.note

    async def _bg():
        await asyncio.to_thread(_resume_workflow_sync, doc_id, decision, feedback_note)

    background_tasks.add_task(_bg)

    return {"document_id": doc_id, "status": next_status}


@app.delete("/documents/{document_id}", status_code=200)
async def delete_document(
    document_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    # Verify owner of document first
    result = (
        get_supabase()
        .table("documents")
        .select("id")
        .eq("id", str(document_id))
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found or unauthorized")

    # Delete the document (cascade will delete revisions and agent_runs)
    get_supabase().table("documents").delete().eq("id", str(document_id)).execute()
    
    # Also clean up memory thread if it exists
    _workflow_threads.pop(str(document_id), None)
    
    return {"status": "deleted", "id": str(document_id)}



# PDF Export

def clean_pdf_text(text: str) -> str:
    replacements = {
        "\u2022": "-",      # bullet point to hyphen
        "\u2013": "-",      # en-dash to hyphen
        "\u2014": " -- ",   # em-dash
        "\u2018": "'",      # single quotes
        "\u2019": "'",
        "\u201c": '"',      # double quotes
        "\u201d": '"',
        "\u2026": "...",    # ellipsis
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_inline_text(text: str) -> str:
    """Return safely escaped text for ReportLab paragraphs."""
    from xml.sax.saxutils import escape

    text = clean_pdf_text(text or "")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return escape(text)


def _pdf_content_blocks(content: str):
    heading = re.compile(r"^(#{1,3})\s+(.+)$")
    bullet = re.compile(r"^[-*]\s+(.+)$")
    paragraph_lines: list[str] = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            yield "paragraph", " ".join(paragraph_lines)
            paragraph_lines = []

    for raw_line in (content or "").splitlines():
        line = raw_line.strip()

        if not line:
            yield from flush_paragraph()
            continue

        heading_match = heading.match(line)
        bullet_match = bullet.match(line)

        if heading_match:
            yield from flush_paragraph()
            yield f"h{len(heading_match.group(1))}", heading_match.group(2)
        elif bullet_match:
            yield from flush_paragraph()
            yield "bullet", bullet_match.group(1)
        else:
            paragraph_lines.append(line)

    yield from flush_paragraph()


def build_document_pdf_bytes(title: str, content: str, status: str) -> bytes:
    """Build a readable, wrapped PDF for a generated document."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab not installed. Run: pip install reportlab")

    buffer = BytesIO()
    page_width, _page_height = A4
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=clean_pdf_text(title or "Untitled Document"),
        author="InkGraph",
    )

    ink = colors.HexColor("#2B2622")
    paper = colors.HexColor("#EDE6D6")
    brass = colors.HexColor("#8C7A4B")
    ribbon = colors.HexColor("#A63D34")
    muted = colors.HexColor("#8B8478")
    dark = colors.HexColor("#2B2622")

    styles = {
        "header_brand": ParagraphStyle(
            "InkGraphHeaderBrand",
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=19,
            textColor=paper,
        ),
        "header_meta": ParagraphStyle(
            "InkGraphHeaderMeta",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=paper,
        ),
        "status": ParagraphStyle(
            "InkGraphStatus",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_RIGHT,
            textColor=paper,
        ),
        "title": ParagraphStyle(
            "InkGraphTitle",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            textColor=ink,
            spaceAfter=8,
            splitLongWords=1,
            wordWrap="LTR",
        ),
        "h1": ParagraphStyle(
            "InkGraphHeading1",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=ribbon,
            spaceBefore=8,
            spaceAfter=5,
            splitLongWords=1,
            wordWrap="LTR",
        ),
        "h2": ParagraphStyle(
            "InkGraphHeading2",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=brass,
            spaceBefore=7,
            spaceAfter=4,
            splitLongWords=1,
            wordWrap="LTR",
        ),
        "h3": ParagraphStyle(
            "InkGraphHeading3",
            fontName="Helvetica-BoldOblique",
            fontSize=11,
            leading=15,
            textColor=ink,
            spaceBefore=6,
            spaceAfter=3,
            splitLongWords=1,
            wordWrap="LTR",
        ),
        "paragraph": ParagraphStyle(
            "InkGraphParagraph",
            fontName="Helvetica",
            fontSize=10.5,
            leading=15.5,
            textColor=ink,
            spaceAfter=7,
            splitLongWords=1,
            wordWrap="LTR",
        ),
        "bullet": ParagraphStyle(
            "InkGraphBullet",
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=ink,
            leftIndent=14,
            firstLineIndent=0,
            bulletIndent=4,
            spaceAfter=5,
            splitLongWords=1,
            wordWrap="LTR",
        ),
    }

    header = Table(
        [
            [
                Paragraph("INKGRAPH", styles["header_brand"]),
                Paragraph(f"STATUS: {_pdf_inline_text(status or 'UNKNOWN')}", styles["status"]),
            ],
            [
                Paragraph("Multi-Agent Writing System | AI-Generated Document", styles["header_meta"]),
                "",
            ],
        ],
        colWidths=[122 * mm, 52 * mm],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), dark),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story = [
        header,
        Spacer(1, 11 * mm),
        Paragraph(_pdf_inline_text(title or "Untitled Document"), styles["title"]),
        HRFlowable(width="100%", thickness=1.1, color=brass, spaceBefore=2, spaceAfter=10),
    ]

    blocks = list(_pdf_content_blocks(content))
    if not blocks:
        blocks = [("paragraph", "No document content is available yet.")]

    for block_type, text in blocks:
        safe_text = _pdf_inline_text(text)
        if not safe_text:
            continue
        if block_type == "bullet":
            story.append(Paragraph(safe_text, styles["bullet"], bulletText="-"))
        else:
            story.append(Paragraph(safe_text, styles.get(block_type, styles["paragraph"])))

    def draw_footer(canvas, current_doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawCentredString(
            page_width / 2,
            11 * mm,
            f"Generated by InkGraph | inkgraph-frontend.vercel.app | Page {current_doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()


@app.get("/documents/{document_id}/export/pdf")
async def export_pdf(
    document_id: UUID,
    _user_id: str = Depends(get_current_user_id),
):
    """
    Generate and stream a PDF of the document's current content.
    Uses ReportLab for server-side PDF generation.
    """
    result = (
        get_supabase()
        .table("documents")
        .select("*")
        .eq("id", str(document_id))
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = result.data[0]
    title = doc.get("title", "Untitled Document")
    content = doc.get("current_content") or ""
    status = doc.get("status", "unknown").upper()

    pdf_bytes = build_document_pdf_bytes(title, content, status)
    safe_name = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_").lower()
    filename = f"inkgraph_{safe_name or 'document'}.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
