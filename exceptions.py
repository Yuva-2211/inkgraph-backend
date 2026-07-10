"""
InkGraph backend — custom exception classes and error handlers.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class InkGraphError(Exception):
    """Base exception for InkGraph application errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class WorkflowError(InkGraphError):
    """Raised when the LangGraph workflow encounters an unrecoverable error."""
    def __init__(self, document_id: str, detail: str):
        super().__init__(
            f"Workflow failed for document {document_id}: {detail}",
            status_code=500,
        )
        self.document_id = document_id


class DocumentNotFoundError(InkGraphError):
    """Raised when a document cannot be found or doesn't belong to the user."""
    def __init__(self, document_id: str):
        super().__init__(f"Document {document_id} not found", status_code=404)


async def inkgraph_error_handler(request: Request, exc: InkGraphError) -> JSONResponse:
    """Global FastAPI exception handler for InkGraphError subclasses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )
