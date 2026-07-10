# InkGraph Backend

FastAPI backend for InkGraph, deployed on Render.

## Local setup

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://localhost:8000`.

## Render deployment

Connect this repository to Render as a Blueprint or Web Service. The included `render.yaml` builds from the repository root.

If you create a manual Web Service instead of using the Blueprint, set:

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Set these secret environment variables in Render:

- `GROQ_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Optional:

- `ALLOWED_ORIGINS`
- `MAX_CONCURRENT_RUNS`

## API

- `GET /health`
- `GET /documents`
- `POST /documents`
- `GET /documents/{id}`
- `GET /documents/{id}/revisions`
- `POST /documents/{id}/decision`
- `DELETE /documents/{id}`
- `GET /documents/{id}/export/pdf`
