"""Entry point for the rebuilt VAS time-series pipeline."""

import argparse
import csv
import logging
import os
import json
import re
import itertools
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import torch

from vas.config import Config
from vas.indexer import build_index
from vas.dataset import load_sessions, split_by_subject
from vas.models import SiameseResNet
from vas.trainer import run_kfold
from vas.utils import save_config, setup_logging
from vas.visualize import predict_timeseries, save_timeseries_plot

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuilt VAS time-series trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build index for face_data_vas")
    p_index.add_argument("--data-root", default=Config().data_root)
    p_index.add_argument("--index-path", default=Config().index_path)

    p_train = sub.add_parser("train", help="Run K-fold training")
    p_train.add_argument("--data-root", default=Config().data_root)
    p_train.add_argument("--index-path", default=Config().index_path)
    p_train.add_argument("--output-dir", default=None)
    p_train.add_argument("--num-classes", type=int, default=Config().num_classes)
    p_train.add_argument("--clip-sec", type=int, default=Config().clip_sec)
    p_train.add_argument("--n-folds", type=int, default=Config().n_folds)
    p_train.add_argument("--batch-size", type=int, default=Config().batch_size)
    p_train.add_argument("--epochs", type=int, default=Config().epochs)
    p_train.add_argument("--lr", type=float, default=Config().lr)
    p_train.add_argument("--backbone", default=Config().backbone)
    p_train.add_argument("--pretrained", action="store_true")

    # ... other args ...
    p_train.add_argument("--no-pretrained", action="store_true")
    p_train.add_argument("--no-amp", action="store_true")
    p_train.add_argument("--num-workers", type=int, default=Config().num_workers)
    p_train.add_argument("--train-samples-per-group", type=int, default=Config().train_samples_per_group)
    p_train.add_argument("--val-ratio", type=float, default=Config().val_ratio)
    p_train.add_argument("--infer-stride-sec", type=int, default=Config().infer_stride_sec)
    p_train.add_argument("--smooth-window-sec", type=int, default=Config().smooth_window_sec)
    p_train.add_argument("--flip-prob", type=float, default=Config().flip_prob)
    p_train.add_argument("--brightness-jitter", type=float, default=Config().brightness_jitter)
    p_train.add_argument("--contrast-jitter", type=float, default=Config().contrast_jitter)
    p_train.add_argument("--noise-std", type=float, default=Config().noise_std)
    p_train.add_argument("--pin-memory", action="store_true")
    p_train.add_argument("--no-pin-memory", action="store_true")
    p_train.add_argument("--persistent-workers", action="store_true")
    p_train.add_argument("--no-persistent-workers", action="store_true")
    p_train.add_argument("--no-class-weights", action="store_true")
    p_train.add_argument("--no-focal", action="store_true")
    p_train.add_argument("--no-weighted-sampler", action="store_true")
    p_train.add_argument("--gpus", default=Config().gpus)
    p_train.add_argument("--skip-visualize", action="store_true")

    p_viz = sub.add_parser("visualize", help="Run visualization only")
    p_viz.add_argument("--output-dir", required=True)
    p_viz.add_argument("--index-path", default=None)
    p_viz.add_argument("--fold", type=int, default=None, help="1-based fold number")
    p_viz.add_argument("--infer-stride-sec", type=int, default=None)
    p_viz.add_argument("--smooth-window-sec", type=int, default=None)

    p_viz_video = sub.add_parser("visualize-video", help="Run visualization directly from videos")
    p_viz_video.add_argument("--output-dir", required=True)
    p_viz_video.add_argument("--video-root", default="/home/user/alcohol_exp/database")
    p_viz_video.add_argument("--index-path", default=None)
    p_viz_video.add_argument("--fold", type=int, default=None, help="1-based fold number")
    p_viz_video.add_argument("--infer-stride-sec", type=int, default=None)
    p_viz_video.add_argument("--smooth-window-sec", type=int, default=None)

    p_f1 = sub.add_parser("summarize-f1", help="Summarize CV metrics from cv_results.csv")
    p_f1.add_argument("--output-dir", action="append", default=[], help="Run directory (repeatable)")
    p_f1.add_argument("--outputs-root", default=None, help="Scan outputs root for run_* directories")
    p_f1.add_argument("--csv-out", default=None, help="Write summary CSV to this path")

    p_sweep = sub.add_parser("sweep", help="Run a small hyperparameter sweep")
    p_sweep.add_argument("--data-root", default=Config().data_root)
    p_sweep.add_argument("--index-path", default=Config().index_path)
    p_sweep.add_argument("--output-root", default=Config().output_dir)
    p_sweep.add_argument("--num-classes", type=int, default=Config().num_classes)
    p_sweep.add_argument("--clip-sec", type=int, default=Config().clip_sec)
    p_sweep.add_argument("--n-folds", type=int, default=3)
    p_sweep.add_argument("--epochs", type=int, default=10)
    p_sweep.add_argument("--early-stop-patience", type=int, default=5)
    p_sweep.add_argument("--num-workers", type=int, default=Config().num_workers)
    p_sweep.add_argument("--train-samples-per-group", type=int, default=Config().train_samples_per_group)
    p_sweep.add_argument("--val-ratio", type=float, default=Config().val_ratio)
    p_sweep.add_argument("--lrs", default=str(Config().lr))
    p_sweep.add_argument("--batch-sizes", default=str(Config().batch_size))
    p_sweep.add_argument("--focal-gammas", default=str(Config().focal_gamma))
    p_sweep.add_argument("--flip-prob", type=float, default=Config().flip_prob)
    p_sweep.add_argument("--brightness-jitter", type=float, default=Config().brightness_jitter)
    p_sweep.add_argument("--contrast-jitter", type=float, default=Config().contrast_jitter)
    p_sweep.add_argument("--noise-std", type=float, default=Config().noise_std)
    p_sweep.add_argument("--pin-memory", action="store_true")
    p_sweep.add_argument("--no-pin-memory", action="store_true")
    p_sweep.add_argument("--persistent-workers", action="store_true")
    p_sweep.add_argument("--no-persistent-workers", action="store_true")
    p_sweep.add_argument("--no-amp", action="store_true")
    p_sweep.add_argument("--no-class-weights", action="store_true")
    p_sweep.add_argument("--no-focal", action="store_true")
    p_sweep.add_argument("--no-weighted-sampler", action="store_true")
    p_sweep.add_argument("--gpus", default=Config().gpus)
    p_sweep.add_argument("--with-visualize", action="store_true")
    p_sweep.add_argument("--tag", default=None)

    return parser.parse_args()


def ensure_index(data_root: str, index_path: str) -> None:
    if not Path(index_path).exists():
        stats = build_index(data_root, index_path)
        logger.info("Index built: sessions=%s images=%s skipped=%s", stats.sessions, stats.images, stats.skipped)


def load_model(checkpoint_path: Path, cfg: Config, device: torch.device) -> SiameseResNet:
    model = SiameseResNet(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        pretrained=False,
        dropout=cfg.dropout,
    )
    state = torch.load(checkpoint_path, map_location=device)["model"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_visualizations(
    cfg: Config,
    output_dir: str,
    splits: List[Tuple[List[str], List[str], List[str]]],
    only_fold: Optional[int] = None,
) -> None:
    sessions = load_sessions(cfg.index_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (_, _, test_ids) in enumerate(splits):
        if only_fold is not None and (fold_idx + 1) != only_fold:
            continue
        fold_dir = Path(output_dir) / f"fold_{fold_idx}"
        model_path = fold_dir / "model_best.pt"
        if not model_path.exists():
            logger.warning("Missing model for fold %s: %s", fold_idx + 1, model_path)
            continue

        model = load_model(model_path, cfg, device)
        out_dir = fold_dir / "timeseries"
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(test_ids)
        for idx, session_id in enumerate(test_ids, 1):
            session = sessions.get(session_id)
            if session is None:
                continue
            try:
                data = predict_timeseries(model, session, cfg, device)
                save_timeseries_plot(out_dir, session, data, cfg)
                logger.info("Fold %s: %s/%s %s done", fold_idx + 1, idx, total, session_id)
            except Exception as exc:
                logger.warning("Timeseries failed for %s: %s", session_id, exc)


def write_cv_results(results: List[dict], output_dir: str) -> None:
    output_path = Path(output_dir) / "cv_results.csv"
    max_classes = 0
    for row in results:
        cm = row.get("cm")
        if cm:
            max_classes = max(max_classes, len(cm))
    if max_classes == 0:
        for row in results:
            for key in row.keys():
                match = re.match(r"class(\d+)_precision", key)
                if match:
                    max_classes = max(max_classes, int(match.group(1)) + 1)

    fields = ["fold", "accuracy", "macro_f1"]
    for i in range(max_classes):
        fields.append(f"class{i}_precision")
        fields.append(f"class{i}_recall")
        fields.append(f"class{i}_f1")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})


def _mean_std(values: List[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _summarize_dir(output_dir: Path) -> Optional[dict]:
    input_path = output_dir / "cv_results.csv"
    if not input_path.exists():
        print(f"{output_dir.name}: cv_results.csv not found")
        return None

    metrics = {
        "accuracy": [],
        "macro_f1": [],
        "class0_precision": [],
        "class0_recall": [],
        "class0_f1": [],
    }

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            for key in metrics:
                try:
                    metrics[key].append(float(row.get(key, "")))
                except (TypeError, ValueError):
                    pass

    if not metrics["macro_f1"]:
        print(f"{output_dir.name}: no metric values found")
        return None

    summary = {"run": output_dir.name, "folds": len(metrics["macro_f1"])}
    print(f"run: {output_dir.name}")
    print(f"folds: {len(metrics['macro_f1'])}")
    for key, values in metrics.items():
        if not values:
            continue
        mean, std = _mean_std(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        print(f"{key}: mean={mean:.4f} std={std:.4f}")
    return summary


def summarize_f1(output_dirs: List[str], outputs_root: Optional[str], csv_out: Optional[str]) -> None:
    dirs: List[Path] = []
    for output_dir in output_dirs:
        dirs.append(Path(output_dir))

    if outputs_root:
        root = Path(outputs_root)
        if root.exists():
            if (root / "cv_results.csv").exists():
                dirs.append(root)
            for cand in sorted(root.glob("run_*")):
                if (cand / "cv_results.csv").exists():
                    dirs.append(cand)

    if not dirs:
        default_root = Path(Config().output_dir)
        if default_root.exists():
            for cand in sorted(default_root.glob("run_*")):
                if (cand / "cv_results.csv").exists():
                    dirs.append(cand)

    if not dirs:
        raise ValueError("No output directories found. Provide --output-dir or --outputs-root.")

    summaries: List[dict] = []
    for i, output_dir in enumerate(dirs):
        if i > 0:
            print()
        summary = _summarize_dir(output_dir)
        if summary:
            summaries.append(summary)

    if csv_out and summaries:
        output_path = Path(csv_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metric_keys = ["accuracy", "macro_f1", "class0_precision", "class0_recall", "class0_f1"]
        fieldnames = ["run", "folds"]
        for key in metric_keys:
            fieldnames.append(f"{key}_mean")
            fieldnames.append(f"{key}_std")

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                writer.writerow({k: summary.get(k, "") for k in fieldnames})


def _parse_list(value: str, cast) -> List:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def run_sweep(args: argparse.Namespace) -> None:
    lrs = _parse_list(args.lrs, float)
    batch_sizes = _parse_list(args.batch_sizes, int)
    gammas = _parse_list(args.focal_gammas, float)
    if args.no_focal:
        gammas = [Config().focal_gamma]

    pin_memory = Config().pin_memory
    if args.pin_memory:
        pin_memory = True
    if args.no_pin_memory:
        pin_memory = False

    persistent_workers = Config().persistent_workers
    if args.persistent_workers:
        persistent_workers = True
    if args.no_persistent_workers:
        persistent_workers = False

    if not lrs or not batch_sizes or not gammas:
        raise ValueError("Sweep lists cannot be empty. Check --lrs/--batch-sizes/--focal-gammas.")

    ensure_index(args.data_root, args.index_path)

    base_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ""
    if args.tag:
        safe_tag = args.tag.strip().replace(" ", "_")
        tag = f"_{safe_tag}"

    for idx, (lr, batch_size, gamma) in enumerate(itertools.product(lrs, batch_sizes, gammas), 1):
        lr_str = str(lr).replace(".", "p")
        gamma_str = "nofocal" if args.no_focal else str(gamma).replace(".", "p")
        run_name = f"run_{base_stamp}{tag}_sweep{idx:02d}_lr{lr_str}_bs{batch_size}_g{gamma_str}"
        output_dir = Path(args.output_root) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(str(output_dir / "train.log"), force=True)

        cfg = Config(
            data_root=args.data_root,
            index_path=args.index_path,
            output_dir=str(output_dir),
            num_classes=args.num_classes,
            clip_sec=args.clip_sec,
            n_folds=args.n_folds,
            batch_size=batch_size,
            epochs=args.epochs,
            lr=lr,
            train_samples_per_group=args.train_samples_per_group,
            val_ratio=args.val_ratio,
            use_amp=not args.no_amp,
            early_stop_patience=args.early_stop_patience,
            use_class_weights=not args.no_class_weights,
            use_focal_loss=not args.no_focal,
            focal_gamma=gamma,
            use_weighted_sampler=not args.no_weighted_sampler,
            num_workers=args.num_workers,
            gpus=args.gpus,
            flip_prob=args.flip_prob,
            brightness_jitter=args.brightness_jitter,
            contrast_jitter=args.contrast_jitter,
            noise_std=args.noise_std,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        save_config(cfg, str(output_dir))
        logger.info(
            "Sweep %s: lr=%s batch=%s gamma=%s",
            idx,
            lr,
            batch_size,
            "disabled" if args.no_focal else gamma,
        )

        results, splits = run_kfold(cfg, str(output_dir))
        write_cv_results(results, str(output_dir))

        if args.with_visualize:
            run_visualizations(cfg, str(output_dir), splits)


def main() -> None:
    args = parse_args()

    if args.command == "index":
        setup_logging()
        stats = build_index(args.data_root, args.index_path)
        logger.info("Index built: sessions=%s images=%s skipped=%s", stats.sessions, stats.images, stats.skipped)
        return

    if args.command == "train":
        output_dir = args.output_dir
        if not output_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(Path(Config().output_dir) / f"run_{stamp}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        setup_logging(os.path.join(output_dir, "train.log"))

        pretrained = Config().pretrained
        if args.pretrained:
            pretrained = True
        if args.no_pretrained:
            pretrained = False

        pin_memory = Config().pin_memory
        if args.pin_memory:
            pin_memory = True
        if args.no_pin_memory:
            pin_memory = False

        persistent_workers = Config().persistent_workers
        if args.persistent_workers:
            persistent_workers = True
        if args.no_persistent_workers:
            persistent_workers = False

        cfg = Config(
            data_root=args.data_root,
            index_path=args.index_path,
            output_dir=output_dir,
            num_classes=args.num_classes,
            clip_sec=args.clip_sec,
            n_folds=args.n_folds,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            backbone=args.backbone,
            pretrained=pretrained,
            use_amp=not args.no_amp,
            num_workers=args.num_workers,
            train_samples_per_group=args.train_samples_per_group,
            val_ratio=args.val_ratio,
            infer_stride_sec=args.infer_stride_sec,
            smooth_window_sec=args.smooth_window_sec,
            flip_prob=args.flip_prob,
            brightness_jitter=args.brightness_jitter,
            contrast_jitter=args.contrast_jitter,
            noise_std=args.noise_std,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            use_class_weights=not args.no_class_weights,
            use_focal_loss=not args.no_focal,
            use_weighted_sampler=not args.no_weighted_sampler,
            gpus=args.gpus,
        )

        ensure_index(cfg.data_root, cfg.index_path)
        save_config(cfg, output_dir)

        results, splits = run_kfold(cfg, output_dir)
        write_cv_results(results, output_dir)

        if not args.skip_visualize:
            run_visualizations(cfg, output_dir, splits)
        return

    if args.command == "visualize":
        output_dir = args.output_dir
        cfg_path = Path(output_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {output_dir}")
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if args.index_path:
            cfg_data["index_path"] = args.index_path
        if args.infer_stride_sec is not None:
            cfg_data["infer_stride_sec"] = args.infer_stride_sec
        if args.smooth_window_sec is not None:
            cfg_data["smooth_window_sec"] = args.smooth_window_sec
        cfg = Config(**cfg_data)

        splits = split_by_subject(load_sessions(cfg.index_path), cfg.n_folds, cfg.seed, cfg.val_ratio)
        setup_logging(os.path.join(output_dir, "visualize.log"))
        run_visualizations(cfg, output_dir, splits, only_fold=args.fold)
        return

    if args.command == "visualize-video":
        from vas.video_infer import run_video_visualization

        output_dir = args.output_dir
        cfg_path = Path(output_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {output_dir}")
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if args.index_path:
            cfg_data["index_path"] = args.index_path
        if args.infer_stride_sec is not None:
            cfg_data["infer_stride_sec"] = args.infer_stride_sec
        if args.smooth_window_sec is not None:
            cfg_data["smooth_window_sec"] = args.smooth_window_sec
        cfg = Config(**cfg_data)

        sessions = load_sessions(cfg.index_path)
        splits = split_by_subject(sessions, cfg.n_folds, cfg.seed, cfg.val_ratio)
        setup_logging(os.path.join(output_dir, "visualize_video.log"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold_idx, (_, _, test_ids) in enumerate(splits):
            if args.fold is not None and (fold_idx + 1) != args.fold:
                continue
            fold_dir = Path(output_dir) / f"fold_{fold_idx}"
            model_path = fold_dir / "model_best.pt"
            if not model_path.exists():
                logger.warning("Missing model for fold %s: %s", fold_idx + 1, model_path)
                continue
            model = load_model(model_path, cfg, device)
            out_dir = fold_dir / "video_timeseries"
            run_video_visualization(cfg, str(out_dir), sessions, test_ids, args.video_root, model, device)
        return

    if args.command == "summarize-f1":
        summarize_f1(args.output_dir, args.outputs_root, args.csv_out)
        return

    if args.command == "sweep":
        run_sweep(args)
        return


if __name__ == "__main__":
    main()
