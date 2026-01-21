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


def _load_face_detector():
    """Load MediaPipe face detection model."""
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.5,
        )
        return detector
    except ImportError:
        logger.warning("MediaPipe not installed. Install with: pip install mediapipe")
        return None
    except Exception as e:
        logger.warning("Failed to load face detector: %s", e)
        return None


def parse_session_id(session_id: str) -> Optional[VideoInfo]:
    """Parse session ID to extract video information.
    
    Expected format: subjectID_date_condition_trial
    Example: 1_20230101_alcohol_1
    """
    pattern = r"^(\d+)_(\d{8})_(\w+)_(\d+)$"
    match = re.match(pattern, session_id)
    if not match:
        logger.debug("Could not parse session_id: %s", session_id)
        return None
    
    subject_id = int(match.group(1))
    date = match.group(2)
    condition = match.group(3)
    trial = match.group(4)
    
    return VideoInfo(
        session_id=session_id,
        subject_id=subject_id,
        date=date,
        condition=condition,
        trial=trial,
    )


def find_video_file(video_root: str, info: VideoInfo) -> Optional[str]:
    """Search for video file matching the session info."""
    root = Path(video_root)
    if not root.exists():
        return None
    
    # Common video extensions
    extensions = [".mp4", ".avi", ".mov", ".mkv"]
    
    # Try different naming patterns
    patterns = [
        f"{info.session_id}",
        f"{info.subject_id}_{info.date}_{info.condition}_{info.trial}",
        f"subject{info.subject_id}_{info.date}_{info.condition}",
    ]
    
    for pattern in patterns:
        for ext in extensions:
            # Direct match
            candidate = root / f"{pattern}{ext}"
            if candidate.exists():
                return str(candidate)
            
            # Search in subdirectories
            for match in root.rglob(f"*{pattern}*{ext}"):
                return str(match)
    
    return None


def _video_duration(cap: cv2.VideoCapture) -> float:
    """Get video duration in seconds."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps > 0:
        return frame_count / fps
    return 0.0


def _extract_frame(
    cap: cv2.VideoCapture,
    t_sec: float,
    detector,
    transform,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Extract and preprocess a single frame from video at time t_sec."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return None
    
    frame_idx = int(t_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face and crop if detector available
    if detector is not None:
        try:
            results = detector.process(frame_rgb)
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = frame_rgb.shape[:2]
                
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                
                # Add margin
                margin = int(0.2 * max(x2 - x1, y2 - y1))
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                frame_rgb = frame_rgb[y1:y2, x1:x2]
        except Exception as e:
            logger.debug("Face detection failed: %s", e)
    
    # Apply transform
    try:
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        tensor = transform(pil_img)
        return tensor.unsqueeze(0).to(device)
    except Exception as e:
        logger.debug("Transform failed: %s", e)
        return None


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
