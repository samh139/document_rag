from __future__ import annotations

from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent


def load_prompt(category: str, version: str = "v1") -> str:
    path = BASE_PATH / category / f"{category}_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()
