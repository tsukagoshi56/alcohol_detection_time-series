"""Utility helpers."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .config import Config


def setup_logging(log_path: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def save_config(cfg: Config, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
