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
from .dataset import SessionData, SequenceTransform
from .visualize import save_timeseries_plot

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    session_id: str
    subject_id: int
    date: str
    condition: str
    trial: str


SESSION_RE = re.compile(r"^subj(?P<sid>\d+)_(?P<date>\d{8})_(?P<cond>.+)(?P<trial>\d{2})$")


def parse_session_id(session_id: str) -> Optional[VideoInfo]:
    match = SESSION_RE.match(session_id)
    if not match:
        return None
    return VideoInfo(
        session_id=session_id,
        subject_id=int(match.group("sid")),
        date=match.group("date"),
        condition=match.group("cond"),
        trial=match.group("trial"),
    )


def _date_short(date: str) -> str:
    return date[2:] if len(date) == 8 else date


def find_video_file(video_root: str, info: VideoInfo) -> Optional[str]:
    root = Path(video_root)
    if not root.exists():
        return None

    date_short = _date_short(info.date)
    subj_str = str(info.subject_id)

    # Prefer a directory matching date/condition/trial
    pattern_dir = f"{info.date}_*{info.condition}{info.trial}"
    for cand in root.glob(pattern_dir):
        cam_dir = cand / "カメラデータ"
        if cam_dir.exists():
            for mp4 in cam_dir.glob("*.mp4"):
                name = mp4.name
                if subj_str in name and date_short in name:
                    return str(mp4)

    # Fallback: any mp4 containing subject id and date short
    for mp4 in root.rglob("*.mp4"):
        name = mp4.name
        if subj_str in name and date_short in name:
            return str(mp4)

    return None


def _center_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def _load_face_detector():
    try:
        import mediapipe as mp
    except Exception:
        return None
    return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def _detect_face(frame: np.ndarray, detector) -> Optional[np.ndarray]:
    if detector is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if res.detections:
        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
    return None


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _build_sequence(
    cap: cv2.VideoCapture,
    start_sec: float,
    clip_sec: int,
    seq_len: int,
    detector,
    transform: SequenceTransform,
    device: torch.device,
) -> Optional[torch.Tensor]:
    times = np.linspace(start_sec, start_sec + clip_sec, seq_len, endpoint=False)
    frames = []
    for t in times:
        frame = _read_frame_at(cap, t)
        if frame is None:
            continue
        face = _detect_face(frame, detector)
        if face is None:
            continue
        face = cv2.resize(face, transform.size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(face).permute(2, 0, 1))

    if not frames:
        return None

    seq = torch.stack(frames, dim=0)
    seq = transform(seq).unsqueeze(0).to(device)
    return seq


def _video_duration(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return float(total / fps) if total > 0 else 0.0


def run_video_visualization(
    cfg: Config,
    output_dir: str,
    sessions: Dict[str, SessionData],
    session_ids: Iterable[str],
    video_root: str,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    detector = _load_face_detector()
    if detector is None:
        logger.warning("MediaPipe not available; skipping video inference")
        return

    transform = SequenceTransform(cfg, train=False)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_list = list(session_ids)
    for idx, session_id in enumerate(session_list, 1):
        session = sessions.get(session_id)
        if session is None:
            logger.warning("Session not found in index: %s", session_id)
            continue
        info = parse_session_id(session_id)
        if info is None:
            logger.warning("Unable to parse session id: %s", session_id)
            continue
        video_path = find_video_file(video_root, info)
        if not video_path:
            logger.warning("Video not found for %s", session_id)
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Failed to open video: %s", video_path)
            continue

        duration = _video_duration(cap)
        if duration <= cfg.clip_sec:
            cap.release()
            logger.warning("Video too short: %s", video_path)
            continue

        anchor_seq = _build_sequence(
            cap,
            start_sec=0.0,
            clip_sec=cfg.clip_sec,
            seq_len=cfg.seq_len,
            detector=detector,
            transform=transform,
            device=device,
        )
        if anchor_seq is None:
            cap.release()
            logger.warning("Anchor extraction failed: %s", session_id)
            continue

        times = []
        probs = []
        t = 0.0
        model.eval()
        with torch.no_grad():
            while t + cfg.clip_sec <= duration:
                target_seq = _build_sequence(
                    cap,
                    start_sec=t,
                    clip_sec=cfg.clip_sec,
                    seq_len=cfg.seq_len,
                    detector=detector,
                    transform=transform,
                    device=device,
                )
                if target_seq is None:
                    t += cfg.infer_stride_sec
                    continue
                logits = model(anchor_seq, target_seq)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
                times.append(t + cfg.clip_sec / 2.0)
                probs.append(prob)
                t += cfg.infer_stride_sec

        cap.release()

        if probs:
            data = {"times": np.array(times), "probs": np.stack(probs, axis=0)}
            save_timeseries_plot(out_dir, session, data, cfg)
            logger.info("Video viz %s/%s done: %s", idx, len(session_list), session_id)

    if detector is not None:
        detector.close()
