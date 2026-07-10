"""
Node implementations using the Groq LLM API (openai/gpt-oss-120b).

Each node receives the DocumentState dict and returns a partial state update.
All LLM calls are synchronous — run inside asyncio.to_thread in main.py.
"""

import json
import re
import logging

from groq import Groq
from duckduckgo_search import DDGS

from config import settings

logger = logging.getLogger("inkgraph")

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


MODEL = "openai/gpt-oss-120b"


def _chat(messages: list[dict], max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Call Groq chat completion and return the response text."""
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _extract_json(text: str) -> dict:
    """Extract the first JSON object found in a string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def _strip_markdown(text: str) -> str:
    """Remove common Markdown markers and preserve plain text."""
    if not text:
        return ""

    cleaned = text
    cleaned = re.sub(r"(?m)^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^[-*+]\s+", "", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`(.+?)`", r"\1", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned)
    cleaned = re.sub(r"^>\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _enforce_word_limit(text: str, max_words: int | None) -> str:
    if not max_words:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    logger.warning("Content exceeded word limit (%s words); truncating to %s words.", len(words), max_words)
    return " ".join(words[:max_words]).strip()


# Planner Agent


def planner_node(state: dict) -> dict:
    """
    Turn state['prompt'] into a structured JSON outline.
    Returns: {"outline": {"title": str, "sections": [{"heading": str, "key_points": [...]}]}}
    """
    prompt = state["prompt"]

    response = _chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert document planner. Your task is to create a detailed, "
                    "logical outline for a professional document based on a user's prompt.\n\n"
                    "Return ONLY valid JSON in this exact structure:\n"
                    "{\n"
                    '  "title": "Document Title",\n'
                    '  "sections": [\n'
                    '    {\n'
                    '      "heading": "Section Heading",\n'
                    '      "key_points": ["Point 1", "Point 2", "Point 3"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "Include 4-6 sections. Be specific and actionable."
                ),
            },
            {
                "role": "user",
                "content": f"Create a document outline for:\n\n{prompt}",
            },
        ],
        max_tokens=1024,
        temperature=0.5,
    )

    outline = _extract_json(response)
    if not outline or "sections" not in outline:
        outline = {
            "title": "Document Draft",
            "sections": [
                {"heading": "Overview", "key_points": [prompt]},
                {"heading": "Key Points", "key_points": ["Detail 1", "Detail 2"]},
                {"heading": "Conclusion", "key_points": ["Summary"]},
            ],
        }

    return {"outline": outline}



# Search Agent 

def search_node(state: dict) -> dict:
    """
    Formulates a search query from prompt/outline and searches the web via DuckDuckGo.
    Saves results to state['search_results'].
    """
    prompt = state.get("prompt", "")
    outline = state.get("outline") or {}
    title = outline.get("title", prompt[:30])

    # 1. Ask the model to generate a search query
    query_response = _chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search assistant. Based on the document title and prompt, "
                    "generate a single, highly effective search query to find fresh news, facts, "
                    "or background information on the topic. Return only the query string, nothing else."
                ),
            },
            {"role": "user", "content": f"Title: {title}\nPrompt: {prompt}"},
        ],
        max_tokens=100,
        temperature=0.3,
    )
    query = query_response.replace('"', '').strip()
    if not query:
        query = title

    # 2. Perform search using duckduckgo_search
    results_text = ""
    try:
        with DDGS() as ddgs:
            # First try news
            news_results = list(ddgs.news(query, max_results=3))
            if news_results:
                for idx, r in enumerate(news_results):
                    results_text += f"[{idx+1}] TITLE: {r.get('title')}\nURL: {r.get('url')}\nSUMMARY: {r.get('body')}\n\n"
            else:
                # Fallback to text search
                text_results = list(ddgs.text(query, max_results=3))
                for idx, r in enumerate(text_results):
                    results_text += f"[{idx+1}] TITLE: {r.get('title')}\nURL: {r.get('url')}\nSUMMARY: {r.get('body')}\n\n"
    except Exception as e:
        logger.error(f"Search failed: {e}")
        results_text = "No search results available (offline or search rate limit)."

    if not results_text.strip():
        results_text = "No relevant search results found."

    return {"search_results": results_text}



# Writer Agent


def writer_node(state: dict) -> dict:
    """
    Generates/revises a draft based on the outline, search_results, style choice, and word limit.
    """
    outline = state.get("outline") or {}
    review_notes = state.get("review_notes") or []
    review_cycle = state.get("review_cycle", 0)
    search_results = state.get("search_results") or "No search results available."
    writing_style = state.get("writing_style") or "general"
    word_limit = state.get("word_limit")

    # Mapping styles to clear instructions
    style_prompts = {
        "explanatory": "Write with a teaching tone. Explain concepts, details, and context clearly.",
        "concise": "Be brief and straight to the point. Eliminate fluff and passive sentences.",
        "technical": "Use technical vocabulary, diagrams description, and precise specs. Write for developers/engineers.",
        "academic": "Write in a formal, scholarly tone. Avoid personal pronouns and casual speech. Be analytical.",
        "general": "Write in a professional, clear, and readable style suitable for any audience.",
    }
    style_instruction = style_prompts.get(writing_style, style_prompts["general"])

    word_limit_instruction = ""
    if word_limit:
        word_limit_instruction = f"Strict word limit: Do NOT exceed {word_limit} words. Keep your content within this limit."

    outline_text = json.dumps(outline, indent=2)

    system_prompt = (
        "You are an expert document writer styled like Claude. You produce highly structured, "
        "coherent, and engaging documents in plain text only. Do NOT use markdown headings, "
        "bold markers, list bullets, or any markdown syntax. Use only paragraphs and sentences.\n\n"
        f"Writing Mode: {style_instruction}\n"
        f"{word_limit_instruction}\n"
        "Always obey the requested writing style and word limit exactly."
    )

    if review_notes and review_cycle > 0:
        recent_notes = "\n".join(f"  • {note}" for note in review_notes[-3:])
        user_msg = (
            f"Update the document draft based on this outline:\n{outline_text}\n\n"
            f"Here are the web search facts to include or reference:\n{search_results}\n\n"
            f"You MUST address these feedback notes in your update:\n{recent_notes}\n\n"
            "Produce the full updated document."
        )
    else:
        user_msg = (
            f"Write a complete document based on this outline:\n{outline_text}\n\n"
            f"Here are the fresh web search facts to support your writing:\n{search_results}\n\n"
            "Write the complete document content now."
        )

    draft = _chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=2048,
        temperature=0.7,
    )

    draft = _strip_markdown(draft)
    draft = _enforce_word_limit(draft, word_limit)
    return {"draft": draft}



# Fact-Checker Agent (NEW)


def fact_checker_node(state: dict) -> dict:
    """
    Compares the writer's draft against the search results.
    If factual errors or major hallucinations are found, it sets needs_revision=True.
    """
    draft = state.get("draft") or ""
    search_results = state.get("search_results") or ""
    review_notes = list(state.get("review_notes") or [])
    review_cycle = state.get("review_cycle", 0)

    if not search_results or "No search results" in search_results:
        # Skip check if no search results were available
        return {"needs_revision": False}

    response = _chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict AI Fact Checker. Compare the written document against "
                    "the provided web search results. Identify any claims that are contradicted "
                    "by the search results or present severe factual errors.\n\n"
                    "Return ONLY valid JSON:\n"
                    "{\n"
                    '  "has_factual_errors": <true or false>,\n'
                    '  "errors_found": ["error detail 1", "error detail 2"],\n'
                    '  "corrections": "What the writer should change to make it factually correct"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": f"DOCUMENT DRAFT:\n{draft}\n\nWEB SEARCH RESULTS:\n{search_results}",
            },
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    result = _extract_json(response)
    has_errors = result.get("has_factual_errors", False)
    errors = result.get("errors_found") or []
    corrections = result.get("corrections", "")

    fact_check_report = ""
    if has_errors and errors:
        fact_check_report = f"[Fact Checker Notice] Factual errors found: {', '.join(errors)}. Corrections needed: {corrections}"
        # If errors found, let's request revision (up to 2 cycles max)
        if review_cycle < 2:
            review_notes.append(fact_check_report)
            return {
                "needs_revision": True,
                "review_notes": review_notes,
            }

    return {
        "needs_revision": False,
    }



# Reviewer Agent


def reviewer_node(state: dict) -> dict:
    """
    Score state['draft'] on grammar, clarity, and completeness.
    Sets state['needs_revision'] and appends to state['review_notes'].
    Caps revision cycles at 2 to prevent infinite loops.
    """
    draft = state.get("draft") or ""
    review_cycle = state.get("review_cycle", 0)
    review_notes = list(state.get("review_notes") or [])
    max_cycles = 2

    # Hard cap on revision cycles
    if review_cycle >= max_cycles:
        return {
            "needs_revision": False,
            "review_notes": review_notes,
            "review_cycle": review_cycle + 1,
        }

    response = _chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior professional document reviewer. Evaluate documents on:\n"
                    "1. Grammar & spelling\n"
                    "2. Clarity & readability\n"
                    "3. Completeness & depth\n"
                    "4. Professional tone\n"
                    "5. Structure & flow\n\n"
                    "Return ONLY valid JSON:\n"
                    "{\n"
                    '  "score": <integer 1-10>,\n'
                    '  "needs_revision": <true if score < 8, false otherwise>,\n'
                    '  "feedback": "<specific, actionable feedback in 2-3 sentences>",\n'
                    '  "strengths": "<what works well>"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": f"Review this document:\n\n{draft}",
            },
        ],
        max_tokens=512,
        temperature=0.3,
    )

    review = _extract_json(response)
    needs_revision = review.get("needs_revision", False)
    feedback = review.get("feedback", "Document meets quality standards.")
    score = review.get("score", 8)

    review_notes.append(f"[Score: {score}/10] {feedback}")

    return {
        "needs_revision": bool(needs_revision),
        "review_notes": review_notes,
        "review_cycle": review_cycle + 1,
    }



# Tone & Style Optimizer Agent


def tone_optimizer_node(state: dict) -> dict:
    """
    Polish the approved draft: improve word choice, sentence rhythm,
    tone consistency, and overall readability without changing content.
    """
    draft = state.get("draft") or ""
    word_limit = state.get("word_limit")
    writing_style = state.get("writing_style") or "general"

    if not draft.strip():
        logger.warning("Tone optimizer received an empty draft; skipping polish step.")
        return {"draft": draft}

    style_prompts = {
        "explanatory": "Use a teaching tone that explains concepts clearly and guides the reader.",
        "concise": "Use concise, direct language with no extra filler.",
        "technical": "Use precise technical terminology and a professional engineering voice.",
        "academic": "Use formal, analytical language appropriate for scholarly writing.",
        "general": "Use a professional, clear, and readable style suitable for any audience.",
    }
    style_instruction = style_prompts.get(writing_style, style_prompts["general"])

    optimized = _chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional editor and writing coach specializing in tone and style. "
                    "Your job is to polish documents to make them shine — improve word choice, "
                    "sentence rhythm, paragraph flow, and tone consistency.\n\n"
                    "Rules:\n"
                    "- Keep ALL the same content, facts, and structure\n"
                    "- Improve phrasing without changing meaning\n"
                    "- Ensure consistent professional tone throughout\n"
                    "- Vary sentence length for better rhythm\n"
                    "- Eliminate redundancy and passive voice\n"
                    "- Return plain text only. Do NOT use markdown headings, bullets, or bold markers.\n"
                    "- Obey the requested writing style and word limit exactly.\n"
                    "Return only the polished document, no commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Optimize the tone and style of this document in {writing_style} style and "
                    f"do not exceed {word_limit} words if a word limit is set:\n\n{draft}"
                    if word_limit
                    else f"Optimize the tone and style of this document in {writing_style} style:\n\n{draft}"
                ),
            },
        ],
        max_tokens=2048,
        temperature=0.5,
    )

    optimized = _strip_markdown(optimized)
    optimized = _enforce_word_limit(optimized, word_limit)
    return {"draft": optimized}



# Human Gate (no-op interrupt point)


def human_gate_node(state: dict) -> dict:
    """
    No-op pass-through. This node exists purely as the LangGraph interrupt point.
    The actual human decision arrives via POST /documents/{id}/decision,
    which calls workflow_app.update_state() and resumes the graph.
    """
    return {"human_decision": state.get("human_decision")}
