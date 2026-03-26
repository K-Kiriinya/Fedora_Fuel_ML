# This helps in avoiding hard coding paths.

# from __future__ import annotations
# import os
from pathlib import Path

# BASE_DIR resolves to project root automatically
BASE_DIR = Path(__file__).resolve().parents[3]

# Data directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = DATA_DIR / "runs"

# Ensure folders exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
