# InkGraph Backend

FastAPI backend for InkGraph. This service runs the multi-agent document workflow, stores document state in Supabase, exports approved drafts as PDFs, and exposes the API consumed by the Vercel frontend.

Production backend URL:

```text
https://inkgraph-backend.onrender.com
```

Frontend origin allowed by default:

```text
https://inkgraph-frontend.vercel.app
```

## Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- Supabase Auth and PostgreSQL
- LangGraph
- Groq API
- DuckDuckGo search
- ReportLab PDF generation

## Repository Structure

```text
.
+-- main.py              # FastAPI app, routes, workflow runners, PDF export
+-- auth.py              # Supabase JWT auth dependency
+-- config.py            # Environment variable settings
+-- db.py                # Supabase client
+-- exceptions.py        # Shared error helpers
+-- rate_limit.py        # Concurrency/rate-limit helpers
+-- version.py           # Health endpoint metadata
+-- graph/
|   +-- workflow.py      # LangGraph workflow definition
|   +-- nodes.py         # Agent node implementations
+-- supabase/schema.sql  # Required database schema
+-- requirements.txt
+-- render.yaml
+-- Procfile
```

## Local Setup

Create and fill the backend environment file:

```bash
cp .env.example .env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn main:app --reload
```

Local API URL:

```text
http://localhost:8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

## Environment Variables

Required:

| Variable | Description |
| --- | --- |
| `SUPABASE_URL` | Supabase project URL. |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role key. Server-only. Never expose this in frontend code. |
| `GROQ_API_KEY` | Groq API key used by the agent workflow. |

Recommended:

| Variable | Description |
| --- | --- |
| `ALLOWED_ORIGINS` | JSON array of allowed browser origins for CORS. |
| `MAX_CONCURRENT_RUNS` | Maximum simultaneous workflow runs. Defaults to `20`. |

Example:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
GROQ_API_KEY=your-groq-api-key
ALLOWED_ORIGINS=["http://localhost:5173","https://inkgraph-frontend.vercel.app"]
MAX_CONCURRENT_RUNS=20
```

## Database Setup

Run the SQL in:

```text
supabase/schema.sql
```

The backend expects these tables:

- `documents`
- `revisions`

Supabase Auth is used for users. API requests must include a valid Supabase access token:

```http
Authorization: Bearer <supabase-access-token>
```

## Render Deployment

This repository is configured for Render.

### Blueprint Deploy

Use `render.yaml` from the repo root. It defines:

```bash
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
Health Check Path: /health
```

### Manual Web Service Deploy

If creating a Render Web Service manually, use:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set these Render environment variables:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
GROQ_API_KEY
ALLOWED_ORIGINS=["https://inkgraph-frontend.vercel.app","http://localhost:5173"]
MAX_CONCURRENT_RUNS=20
```

After dependency changes, deploy with:

```text
Manual Deploy -> Clear build cache & deploy
```

## API Reference

Health:

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | No | Returns API status and version metadata. |

Documents:

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/documents` | Yes | List documents for the current user. |
| `POST` | `/documents` | Yes | Create a document and start the agent workflow. |
| `GET` | `/documents/{document_id}` | Yes | Get one document. |
| `GET` | `/documents/{document_id}/revisions` | Yes | Get workflow revision history. |
| `POST` | `/documents/{document_id}/decision` | Yes | Approve or request changes after human review. |
| `DELETE` | `/documents/{document_id}` | Yes | Delete a document. |
| `GET` | `/documents/{document_id}/export/pdf` | Yes | Export current document content as a PDF. |

Create document body:

```json
{
  "title": "Market Research Brief",
  "prompt": "Write a market research brief for...",
  "word_limit": 1200,
  "writing_style": "general"
}
```

Human decision body:

```json
{
  "decision": "approved",
  "note": "Looks good."
}
```

Allowed `decision` values:

- `approved`
- `changes`

## Agent Workflow

The workflow is defined in `graph/workflow.py`.

```text
planner -> search -> writer -> fact_checker -> reviewer -> tone_optimizer -> human review
```

If review or fact-checking requests changes, the workflow loops back to the writer. When the human approves, the document status becomes `approved`.

## PDF Export

PDF export uses ReportLab. The endpoint:

```text
GET /documents/{document_id}/export/pdf
```

returns a binary PDF response. The frontend downloads it using `fetch()` with the Supabase bearer token.

## Troubleshooting

### Backend is offline

Check:

```text
https://inkgraph-backend.onrender.com/health
```

Expected result:

```json
{
  "name": "InkGraph API",
  "status": "ok"
}
```

### Browser shows `Failed to fetch`

Most common causes:

- `VITE_API_BASE_URL` is wrong in Vercel.
- `ALLOWED_ORIGINS` does not include the frontend origin.
- Backend is not deployed or is sleeping.

Correct frontend origin:

```text
https://inkgraph-frontend.vercel.app
```

Correct backend base URL:

```text
https://inkgraph-backend.onrender.com
```

### Render runs the wrong command

Use this exact start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Do not use:

```bash
gunicorn app:app
```

The FastAPI app is in `main.py`, not `app.py`.
