"""Dataset and sampling utilities for Siamese VAS classification."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from .config import Config
from .indexer import IndexRecord, load_index


@dataclass
class Frame:
    time_sec: int
    path: str


@dataclass
class VasGroup:
    group_id: str
    vas_value: int
    vas_time_min: Optional[int]
    frames: List[Frame]


@dataclass
class SessionData:
    session_id: str
    subject_id: int
    anchor_frames: List[Frame]
    normal_target_frames: List[Frame]
    vas_groups: Dict[str, VasGroup]
    all_frames: List[Frame]


@dataclass
class Sample:
    session_id: str
    group_id: str
    label: int


class SequenceTransform:
    def __init__(self, cfg: Config, train: bool):
        self.train = train
        self.size = cfg.img_size
        self.mean = torch.tensor(cfg.mean).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.std).view(1, 3, 1, 1)
        self.flip_prob = cfg.flip_prob
        self.brightness_jitter = cfg.brightness_jitter
        self.contrast_jitter = cfg.contrast_jitter
        self.noise_std = cfg.noise_std

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (T, C, H, W), uint8
        seq = seq.float() / 255.0
        if self.train and self.flip_prob > 0 and random.random() < self.flip_prob:
            seq = torch.flip(seq, dims=[3])
        if self.train and self.brightness_jitter > 0:
            factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            seq = seq * factor
        if self.train and self.contrast_jitter > 0:
            factor = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            mean = seq.mean(dim=(0, 2, 3), keepdim=True)
            seq = (seq - mean) * factor + mean
        if self.train and self.noise_std > 0:
            seq = seq + torch.randn_like(seq) * self.noise_std
        if self.train:
            seq = torch.clamp(seq, 0.0, 1.0)
        seq = F.interpolate(seq, size=self.size, mode="bilinear", align_corners=False)
        seq = (seq - self.mean) / self.std
        return seq


def quantize_vas(vas_value: int, num_classes: int) -> int:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if vas_value <= 0:
        return 0
    bin_size = 100.0 / (num_classes - 1)
    idx = int((vas_value - 1) / bin_size) + 1
    return min(idx, num_classes - 1)


def _select_frames(frames: List[Frame], start_sec: float, clip_sec: int) -> List[Frame]:
    end_sec = start_sec + clip_sec
    selected = [f for f in frames if start_sec <= f.time_sec < end_sec]
    if not selected:
        selected = frames[:]
    return selected


def _sample_sequence(frames: List[Frame], seq_len: int) -> List[Frame]:
    if len(frames) >= seq_len:
        idxs = torch.linspace(0, len(frames) - 1, seq_len).long().tolist()
        return [frames[i] for i in idxs]
    if not frames:
        return []
    pad = [frames[-1]] * (seq_len - len(frames))
    return frames + pad


def _pick_start(frames: List[Frame], clip_sec: int, mode: str) -> float:
    if not frames:
        return 0.0
    times = [f.time_sec for f in frames]
    min_t = min(times)
    max_t = max(times)
    if max_t - min_t <= clip_sec:
        return float(min_t)
    if mode == "random":
        return random.uniform(min_t, max_t - clip_sec)
    return float(min_t + (max_t - min_t - clip_sec) / 2.0)


def load_sessions(index_path: str) -> Dict[str, SessionData]:
    sessions: Dict[str, SessionData] = {}
    for record in load_index(index_path):
        session = sessions.get(record.session_id)
        if session is None:
            session = SessionData(
                session_id=record.session_id,
                subject_id=record.subject_id,
                anchor_frames=[],
                normal_target_frames=[],
                vas_groups={},
                all_frames=[],
            )
            sessions[record.session_id] = session

        frame = Frame(time_sec=record.time_sec, path=record.path)
        session.all_frames.append(frame)

        if record.label_type == "normal":
            if record.time_sec < 0:
                continue
            if record.time_sec < 300:
                session.anchor_frames.append(frame)
            elif 300 <= record.time_sec < 600:
                session.normal_target_frames.append(frame)
        elif record.label_type == "vas":
            group_id = record.label
            group = session.vas_groups.get(group_id)
            if group is None:
                if record.vas_value is None:
                    continue
                group = VasGroup(
                    group_id=group_id,
                    vas_value=record.vas_value,
                    vas_time_min=record.vas_time_min,
                    frames=[],
                )
                session.vas_groups[group_id] = group
            group.frames.append(frame)

    # Sort frames by time for deterministic sampling
    for session in sessions.values():
        session.anchor_frames.sort(key=lambda f: f.time_sec)
        session.normal_target_frames.sort(key=lambda f: f.time_sec)
        session.all_frames.sort(key=lambda f: f.time_sec)
        for group in session.vas_groups.values():
            group.frames.sort(key=lambda f: f.time_sec)

    return sessions


def split_by_subject(
    sessions: Dict[str, SessionData],
    n_folds: int,
    seed: int,
    val_ratio: float,
) -> List[Tuple[List[str], List[str], List[str]]]:
    subject_to_sessions: Dict[int, List[str]] = {}
    for session_id, session in sessions.items():
        subject_to_sessions.setdefault(session.subject_id, []).append(session_id)

    subjects = list(subject_to_sessions.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    folds = [subjects[i::n_folds] for i in range(n_folds)]
    split_sets: List[Tuple[List[str], List[str], List[str]]] = []

    for fold_idx in range(n_folds):
        test_subjects = folds[fold_idx]
        train_subjects = [s for i, f in enumerate(folds) if i != fold_idx for s in f]

        rng_fold = random.Random(seed + fold_idx)
        rng_fold.shuffle(train_subjects)
        val_count = max(1, int(len(train_subjects) * val_ratio))
        val_subjects = train_subjects[:val_count]
        train_subjects = train_subjects[val_count:]

        def _collect(subj_ids: List[int]) -> List[str]:
            session_ids: List[str] = []
            for sid in subj_ids:
                session_ids.extend(subject_to_sessions[sid])
            return session_ids

        split_sets.append((_collect(train_subjects), _collect(val_subjects), _collect(test_subjects)))

    return split_sets


class SiameseVasDataset(Dataset):
    def __init__(
        self,
        sessions: Dict[str, SessionData],
        session_ids: Sequence[str],
        cfg: Config,
        split: str,
    ):
        self.sessions = sessions
        self.cfg = cfg
        self.split = split
        self.transform = SequenceTransform(cfg, train=(split == "train"))

        samples_per_group = {
            "train": cfg.train_samples_per_group,
            "val": cfg.val_samples_per_group,
            "test": cfg.test_samples_per_group,
        }[split]

        self.samples: List[Sample] = []
        for session_id in session_ids:
            session = sessions[session_id]
            if not session.anchor_frames:
                continue
            if not session.vas_groups and not session.normal_target_frames:
                continue
            for group_id, group in session.vas_groups.items():
                label = quantize_vas(group.vas_value, cfg.num_classes)
                for _ in range(samples_per_group):
                    self.samples.append(Sample(session_id=session_id, group_id=group_id, label=label))

            # Add VAS=0 group from 5-10 min normal window
            if len(session.normal_target_frames) >= cfg.min_frames_per_window:
                group_id = "vas0"
                group = VasGroup(
                    group_id=group_id,
                    vas_value=0,
                    vas_time_min=10,
                    frames=session.normal_target_frames,
                )
                session.vas_groups[group_id] = group
                for _ in range(samples_per_group):
                    self.samples.append(Sample(session_id=session_id, group_id=group_id, label=0))

        if not self.samples:
            raise ValueError(f"No samples for split={split}. Check data/index and filters.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sequence(self, frames: List[Frame]) -> torch.Tensor:
        imgs = []
        for f in frames:
            img = read_image(f.path)
            if img.shape[0] == 4:
                img = img[:3]
            imgs.append(img)
        seq = torch.stack(imgs, dim=0)
        return self.transform(seq)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        session = self.sessions[sample.session_id]
        group = session.vas_groups[sample.group_id]

        anchor_mode = "random" if self.split == "train" else "center"
        target_mode = "random" if self.split == "train" else "center"

        anchor_start = _pick_start(session.anchor_frames, self.cfg.clip_sec, anchor_mode)
        anchor_frames = _select_frames(session.anchor_frames, anchor_start, self.cfg.clip_sec)
        anchor_frames = _sample_sequence(anchor_frames, self.cfg.seq_len)

        target_start = _pick_start(group.frames, self.cfg.clip_sec, target_mode)
        target_frames = _select_frames(group.frames, target_start, self.cfg.clip_sec)
        target_frames = _sample_sequence(target_frames, self.cfg.seq_len)

        if not anchor_frames or not target_frames:
            raise RuntimeError("Empty frame sequence after sampling")

        anchor_seq = self._load_sequence(anchor_frames)
        target_seq = self._load_sequence(target_frames)

        meta = {
            "session_id": session.session_id,
            "subject_id": session.subject_id,
            "group_id": group.group_id,
            "vas_value": group.vas_value,
            "vas_time_min": group.vas_time_min,
        }

        return anchor_seq, target_seq, sample.label, meta


def session_time_windows(
    session: SessionData,
    clip_sec: int,
    stride_sec: int,
    min_frames: int,
) -> List[float]:
    if not session.all_frames:
        return []
    times = [f.time_sec for f in session.all_frames]
    min_t = min(times)
    max_t = max(times)
    starts = []
    t = min_t
    while t <= max_t - clip_sec:
        selected = [f for f in session.all_frames if t <= f.time_sec < t + clip_sec]
        if len(selected) >= min_frames:
            starts.append(float(t))
        t += stride_sec
    return starts


def load_sequence_from_window(
    session: SessionData,
    start_sec: float,
    clip_sec: int,
    seq_len: int,
) -> List[Frame]:
    frames = _select_frames(session.all_frames, start_sec, clip_sec)
    return _sample_sequence(frames, seq_len)
