"""Training and evaluation loops for the Siamese VAS model."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import Config
from .dataset import SiameseVasDataset, split_by_subject, load_sessions
from .metrics import confusion_matrix, classification_report
from .models import SiameseResNetGRU

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = np.zeros_like(counts)
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = total / (num_classes * c)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        targets = targets.view(-1, 1)
        logp_t = logp.gather(1, targets).squeeze(1)
        p_t = p.gather(1, targets).squeeze(1)
        loss = -(1 - p_t) ** self.gamma * logp_t
        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w[targets.squeeze(1)]
        return loss.mean()


def build_dataloader(dataset: SiameseVasDataset, cfg: Config, shuffle: bool) -> DataLoader:
    if shuffle and cfg.use_weighted_sampler:
        labels = [s.label for s in dataset.samples]
        weights = compute_class_weights(labels, cfg.num_classes)
        sample_weights = [weights[l].item() for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=_worker_init,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for anchor, target, labels, _ in pbar:
        anchor = anchor.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(anchor, target)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        pbar.set_postfix(loss=loss.item())

    return total_loss / max(1, total_batches)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for anchor, target, labels, _ in loader:
            anchor = anchor.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(anchor, target)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds, num_classes)
    report = classification_report(cm)
    report["cm"] = cm.tolist()
    return report


def run_kfold(cfg: Config, output_dir: str) -> Tuple[List[Dict[str, float]], List[Tuple[List[str], List[str], List[str]]]]:
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sessions = load_sessions(cfg.index_path)
    splits = split_by_subject(sessions, cfg.n_folds, cfg.seed, cfg.val_ratio)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and torch.cuda.is_available()

    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(splits):
        logger.info("Fold %s: train=%s val=%s test=%s", fold_idx + 1, len(train_ids), len(val_ids), len(test_ids))

        train_ds = SiameseVasDataset(sessions, train_ids, cfg, split="train")
        val_ds = SiameseVasDataset(sessions, val_ids, cfg, split="val")
        test_ds = SiameseVasDataset(sessions, test_ids, cfg, split="test")

        train_loader = build_dataloader(train_ds, cfg, shuffle=True)
        val_loader = build_dataloader(val_ds, cfg, shuffle=False)
        test_loader = build_dataloader(test_ds, cfg, shuffle=False)

        model = SiameseResNetGRU(
            num_classes=cfg.num_classes,
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout,
        )

        if cfg.gpus:
            device_ids = [int(x) for x in cfg.gpus.split(",")]
        else:
            device_ids = list(range(torch.cuda.device_count()))

        model = model.to(device)
        if device.type == "cuda" and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)

        class_weights = None
        if cfg.use_class_weights:
            labels = [s.label for s in train_ds.samples]
            class_weights = compute_class_weights(labels, cfg.num_classes).to(device)

        if cfg.use_focal_loss:
            criterion = FocalLoss(gamma=cfg.focal_gamma, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_metric = -math.inf
        best_path = output_root / f"fold_{fold_idx}" / "model_best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)

        epochs_no_improve = 0
        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_amp, device)
            val_report = evaluate(model, val_loader, device, cfg.num_classes)
            metric = val_report["macro_f1"]

            logger.info(
                "Fold %s Epoch %s: train_loss=%.4f val_macro_f1=%.4f",
                fold_idx + 1,
                epoch + 1,
                train_loss,
                metric,
            )

            if metric > best_metric:
                best_metric = metric
                epochs_no_improve = 0
                torch.save({
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                }, best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.early_stop_patience:
                    logger.info("Early stop on fold %s", fold_idx + 1)
                    break

        # Load best model for test evaluation
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        test_report = evaluate(model, test_loader, device, cfg.num_classes)
        test_report["fold"] = fold_idx + 1

        results.append(test_report)

        fold_metrics_path = output_root / f"fold_{fold_idx}" / "metrics.json"
        fold_metrics_path.write_text(json.dumps(test_report, indent=2))

    return results, splits
