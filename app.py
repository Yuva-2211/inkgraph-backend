"""Compatibility entrypoint for hosts that default to `gunicorn app:app`.

The primary ASGI app lives in `main.py` and should be served with Uvicorn.
This wrapper lets a WSGI Gunicorn command import and serve it if a platform
keeps an older/default start command.
"""

from a2wsgi import ASGIMiddleware

from main import app as fastapi_app

app = ASGIMiddleware(fastapi_app)
