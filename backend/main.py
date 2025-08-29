#!/usr/bin/env python3
"""
FlashFit AI - Main Entry Point

This module serves as the main entry point for the FlashFit AI backend application.
It imports and configures the FastAPI application from app.py.
"""

import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )