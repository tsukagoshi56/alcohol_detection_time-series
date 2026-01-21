"""Run inference directly from source videos."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import Config
from .dataset import SessionData, ImageTransform
from .visualize import save_timeseries_plot

logger = logging.getLogger(__name__)


import csv
from datetime import datetime

@dataclass
class VideoInfo:
    session_id: str
    subject_id: int
    date: str
    condition: str
    trial: str
    video_path: Optional[str] = None


def load_experiment_csv(csv_path: str) -> Dict[str, str]:
    """Load experiment CSV and return a mapping of session_id -> video_path."""
    mapping = {}
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Experiment CSV not found: %s", csv_path)
        return mapping

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expected columns: session_id, video_path, ...
            # Adapt column names as needed based on the user's CSV structure description
            # "experiment_time_only_sapporo.csv" likely has session_id and file path.
            # Let's assume some flexibility or standard naming.
            # If the user said: "experiment_time_only_sapporo.csv"
            # We'll assume columns "session_id" and "video_path" or similar.
            # Since we don't know the exact columns, let's look for "session_id" and "video_path".
            # Or iterate and try to find logical columns.
            
            # User provided example path: /home/user/alcohol_exp/workspace/vas_detection/experiment_time_only_sapporo.csv
            # We'll assume it has 'session_id' and 'video_path' or 'file_path'
            
            sid = row.get("session_id") or row.get("SessionID")
            vpath = row.get("video_path") or row.get("FilePath") or row.get("path")
            
            if sid and vpath:
                mapping[sid.strip()] = vpath.strip()
    return mapping


def run_video_visualization(
    cfg: Config,
    output_dir: str,
    sessions: Dict[str, SessionData],
    session_ids: Iterable[str],
    video_root: str,
    model: torch.nn.Module,
    device: torch.device,
    experiment_csv: Optional[str] = None,
) -> None:
    detector = _load_face_detector()
    if detector is None:
        logger.warning("MediaPipe not available; skipping video inference")
        return

    csv_mapping = {}
    if experiment_csv:
        csv_mapping = load_experiment_csv(experiment_csv)
        logger.info("Loaded %d video paths from CSV", len(csv_mapping))

    transform = ImageTransform(cfg, train=False)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_list = list(session_ids)
    for idx, session_id in enumerate(session_list, 1):
        session = sessions.get(session_id)
        if session is None:
            logger.warning("Session not found in index: %s", session_id)
            continue
        
        video_path = None
        if session_id in csv_mapping:
            video_path = csv_mapping[session_id]
        else:
            # Fallback to parsing/searching logic
            info = parse_session_id(session_id)
            if info:
                video_path = find_video_file(video_root, info)
        
        if not video_path:
            logger.warning("Video not found for %s", session_id)
            continue

        # Convert relative path in CSV to absolute if needed
        if not Path(video_path).is_absolute() and video_root:
             video_path = str(Path(video_root) / video_path)

        if not Path(video_path).exists():
             logger.warning("Video file does not exist: %s", video_path)
             continue

        logger.info("Processing %s: %s", session_id, video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Failed to open video: %s", video_path)
            continue

        duration = _video_duration(cap)
        if duration <= cfg.clip_sec:
            cap.release()
            logger.warning("Video too short: %s", video_path)
            continue

        anchor_tensor = _extract_frame(
            cap,
            t_sec=cfg.clip_sec / 2.0, # Middle of anchor window
            detector=detector,
            transform=transform,
            device=device,
        )
        if anchor_tensor is None:
            cap.release()
            logger.warning("Anchor extraction failed: %s", session_id)
            continue

        times = []
        probs = []
        t = 0.0
        model.eval()
        
        # Use configurable stride
        stride = cfg.infer_stride_sec
        if stride <= 0:
            stride = 10  # Default fallback if config is 0/invalid

        with torch.no_grad():
            while t + cfg.clip_sec <= duration:
                # Sample middle frame of current window
                mid_t = t + cfg.clip_sec / 2.0
                target_tensor = _extract_frame(
                    cap,
                    t_sec=mid_t,
                    detector=detector,
                    transform=transform,
                    device=device,
                )
                if target_tensor is None:
                    t += stride
                    continue
                logits = model(anchor_tensor, target_tensor)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
                times.append(mid_t)
                probs.append(prob)
                t += stride

        cap.release()

        if probs:
            data = {"times": np.array(times), "probs": np.stack(probs, axis=0)}
            save_timeseries_plot(out_dir, session, data, cfg)
            logger.info("Video viz %s/%s done: %s", idx, len(session_list), session_id)

    if detector is not None:
        detector.close()
