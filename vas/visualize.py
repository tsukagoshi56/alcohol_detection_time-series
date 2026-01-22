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
from .dataset import Frame, SessionData, ImageTransform, load_frame_from_window, session_time_windows

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


def _sample_frame(frames: List[Frame]) -> Frame:
    if not frames:
        raise ValueError("No frames")
    # For visualization/stable inference, pick middle frame
    idx = len(frames) // 2
    return frames[idx]


def predict_timeseries(
    model: torch.nn.Module,
    session: SessionData,
    cfg: Config,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    transform = ImageTransform(cfg, train=False)

    if not session.anchor_frames:
        raise ValueError(f"No anchor frames for session {session.session_id}")

    anchor_start = _pick_start(session.anchor_frames, cfg.clip_sec)
    anchor_candidates = [f for f in session.anchor_frames if anchor_start <= f.time_sec < anchor_start + cfg.clip_sec]
    if not anchor_candidates:
        anchor_candidates = session.anchor_frames[:]
    
    anchor_frame = _sample_frame(anchor_candidates)
    anchor_img = torchvision_read(anchor_frame.path)
    # Add batch dim
    anchor_tensor = transform(anchor_img).unsqueeze(0).to(device)

    # For inference, we slide a window and pick ONE frame to represent that window
    starts = session_time_windows(session, cfg.clip_sec, cfg.infer_stride_sec, cfg.min_frames_per_window)

    times = []
    probs = []

    model.eval()
    with torch.no_grad():
        for start in starts:
            frame = load_frame_from_window(session, start, cfg.clip_sec)
            target_img = torchvision_read(frame.path)
            target_tensor = transform(target_img).unsqueeze(0).to(device)
            
            logits = model(anchor_tensor, target_tensor)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            times.append(start + cfg.clip_sec / 2.0)
            probs.append(prob)

    if not probs:
        raise ValueError(f"No predictions produced for session {session.session_id}")

    probs = np.stack(probs, axis=0)
    times = np.array(times)
    return {"times": times, "probs": probs}


def _collect_vas_points(session: SessionData) -> List[tuple[float, int]]:
    points = []
    for group_id, group in session.vas_groups.items():
        if group_id == "vas0":
            continue
        if group.vas_time_min is None:
            continue
        points.append((group.vas_time_min * 60.0, group.vas_value))
    points.sort(key=lambda x: x[0])
    return points


def save_timeseries_plot(output_dir: Path, session: SessionData, data: Dict[str, np.ndarray], cfg: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = data["times"]
    probs = data["probs"]

    smooth_window = max(1, int(cfg.smooth_window_sec / cfg.infer_stride_sec))
    logger.info("Smoothing plot: window_sec=%d, stride_sec=%d => window_points=%d", 
                cfg.smooth_window_sec, cfg.infer_stride_sec, smooth_window)
    smoothed = np.stack([
        _smooth(probs[:, i], smooth_window) for i in range(probs.shape[1])
    ], axis=1)

    # Shift time axis to start from 0
    offset = 0.0
    if len(times) > 0:
        offset = times[0]
    
    plot_times = times - offset

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(probs.shape[1]):
        ax.plot(plot_times, smoothed[:, i], label=f"class{i}")

    # Avoid non-ASCII glyph issues in titles by keeping it simple.
    ax.set_title("Smoothed probabilities")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", ncol=2)

    vas_points = _collect_vas_points(session)
    for t_sec, v in vas_points:
        # Shift VAS points by the same offset
        t_sec_shifted = t_sec - offset
        # Only plot if within visible range
        if 0 <= t_sec_shifted <= plot_times[-1]:
            ax.axvline(t_sec_shifted, color="gray", alpha=0.35, linewidth=1)
            ax.text(t_sec_shifted, 0.98, f"VAS={v}", rotation=90, va="top", ha="right", fontsize=8, color="gray")

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

    if vas_points:
        vas_path = output_dir / f"{session.session_id}_vas_points.csv"
        with vas_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_sec", "vas_value"])
            for t_sec, v in vas_points:
                writer.writerow([f"{t_sec:.2f}", int(v)])


def torchvision_read(path: str) -> torch.Tensor:
    from torchvision.io import read_image

    img = read_image(path)
    if img.shape[0] == 4:
        img = img[:3]
    return img
