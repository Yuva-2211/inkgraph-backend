"""
InkGraph backend — health check & API versioning utilities.
"""

APP_VERSION = "1.0.0"
APP_NAME = "InkGraph API"


def get_version_info() -> dict:
    """Return version metadata for the health endpoint."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
        "agents": [
            "planner",
            "search",
            "writer",
            "fact_checker",
            "reviewer",
            "tone_optimizer",
        ],
    }
