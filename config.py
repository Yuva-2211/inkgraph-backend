"""Environment configuration. Copy .env.example to .env and fill in values."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str  # server-side only — never ship to the client
    GROQ_API_KEY: str
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173"]
    ALLOWED_ORIGIN_REGEX: str | None = r"https://.*\.vercel\.app"

    # Scaling-related knobs — see scaling.md
    REDIS_URL: str = "redis://localhost:6379/0"
    MAX_CONCURRENT_RUNS: int = 20

    class Config:
        env_file = ".env"


settings = Settings()
