"""
FastAPI dependency for Supabase JWT authentication.

Usage:
    @app.get("/some-route")
    async def route(user_id: str = Depends(get_current_user_id)):
        ...
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from db import get_supabase

_bearer = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """
    Verifies the Bearer JWT from the Authorization header via Supabase Auth.
    Returns the authenticated user's UUID string.
    Raises HTTP 401 if the token is missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # Supabase verifies the JWT and returns user info
        response = get_supabase().auth.get_user(token)
        user = response.user
        if not user:
            raise ValueError("No user returned")
        return str(user.id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
