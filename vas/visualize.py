"""Time-series visualization for VAS predictions."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .dataset import Frame, SessionData, SequenceTransform, load_sequence_from_window, session_time_windows

logger = logging.getLogger(__name__)


def _smooth(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(series, kernel, mode="same")


def _pick_start(frames: List[Frame], clip_sec: int) -> float:
    times = [f.time_sec for f in frames]
    if not times:
        return 0.0
    min_t = min(times)
    max_t = max(times)
    if max_t - min_t <= clip_sec:
        return float(min_t)
    return float(min_t + (max_t - min_t - clip_sec) / 2.0)


def _sample_sequence(frames: List[Frame], seq_len: int) -> List[Frame]:
    if len(frames) >= seq_len:
        idxs = np.linspace(0, len(frames) - 1, seq_len).astype(int).tolist()
        return [frames[i] for i in idxs]
    if not frames:
        return []
    pad = [frames[-1]] * (seq_len - len(frames))
    return frames + pad


def predict_timeseries(
    model: torch.nn.Module,
    session: SessionData,
    cfg: Config,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    transform = SequenceTransform(cfg, train=False)

    if not session.anchor_frames:
        raise ValueError(f"No anchor frames for session {session.session_id}")

    anchor_start = _pick_start(session.anchor_frames, cfg.clip_sec)
    anchor_frames = [f for f in session.anchor_frames if anchor_start <= f.time_sec < anchor_start + cfg.clip_sec]
    if not anchor_frames:
        anchor_frames = session.anchor_frames[:]
    anchor_frames = _sample_sequence(anchor_frames, cfg.seq_len)
    anchor_seq = torch.stack([torchvision_read(f.path) for f in anchor_frames], dim=0)
    anchor_seq = transform(anchor_seq).unsqueeze(0).to(device)

    starts = session_time_windows(session, cfg.clip_sec, cfg.infer_stride_sec, cfg.min_frames_per_window)

    times = []
    probs = []

    model.eval()
    with torch.no_grad():
        for start in starts:
            frames = load_sequence_from_window(session, start, cfg.clip_sec, cfg.seq_len)
            if not frames:
                continue
            target_seq = torch.stack([torchvision_read(f.path) for f in frames], dim=0)
            target_seq = transform(target_seq).unsqueeze(0).to(device)
            logits = model(anchor_seq, target_seq)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            times.append(start + cfg.clip_sec / 2.0)
            probs.append(prob)

    if not probs:
        raise ValueError(f"No predictions produced for session {session.session_id}")

    probs = np.stack(probs, axis=0)
    times = np.array(times)
    return {"times": times, "probs": probs}


def save_timeseries_plot(output_dir: Path, session: SessionData, data: Dict[str, np.ndarray], cfg: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = data["times"]
    probs = data["probs"]

    smooth_window = max(1, int(cfg.smooth_window_sec / cfg.infer_stride_sec))
    smoothed = np.stack([
        _smooth(probs[:, i], smooth_window) for i in range(probs.shape[1])
    ], axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(probs.shape[1]):
        ax.plot(times, smoothed[:, i], label=f"class{i}")

    ax.set_title(f"Session {session.session_id} - Smoothed probabilities")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()

    fig_path = output_dir / f"{session.session_id}_timeseries.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    csv_path = output_dir / f"{session.session_id}_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_sec"] + [f"prob_class{i}" for i in range(probs.shape[1])]
        writer.writerow(header)
        for t, row in zip(times, probs):
            writer.writerow([f"{t:.2f}", *[f"{v:.6f}" for v in row]])


def torchvision_read(path: str) -> torch.Tensor:
    from torchvision.io import read_image

    img = read_image(path)
    if img.shape[0] == 4:
        img = img[:3]
    return img
