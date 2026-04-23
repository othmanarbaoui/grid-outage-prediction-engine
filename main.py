"""
main.py
────────
Application entry point.

Run locally
───────────
    python main.py

Or with uvicorn directly (recommended for production):
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
"""

import uvicorn
from app import create_app
from app.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )
