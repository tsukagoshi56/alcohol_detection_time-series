"""Entry point for the rebuilt VAS time-series pipeline."""

import argparse
import csv
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch

from vas.config import Config
from vas.indexer import build_index
from vas.dataset import load_sessions, split_by_subject
from vas.models import SiameseResNetGRU
from vas.trainer import run_kfold
from vas.utils import save_config, setup_logging
from vas.visualize import predict_timeseries, save_timeseries_plot
from vas.video_infer import run_video_visualization

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
    p_train.add_argument("--seq-len", type=int, default=Config().seq_len)
    p_train.add_argument("--n-folds", type=int, default=Config().n_folds)
    p_train.add_argument("--batch-size", type=int, default=Config().batch_size)
    p_train.add_argument("--epochs", type=int, default=Config().epochs)
    p_train.add_argument("--lr", type=float, default=Config().lr)
    p_train.add_argument("--backbone", default=Config().backbone)
    p_train.add_argument("--pretrained", action="store_true")
    p_train.add_argument("--no-pretrained", action="store_true")
    p_train.add_argument("--no-amp", action="store_true")
    p_train.add_argument("--num-workers", type=int, default=Config().num_workers)
    p_train.add_argument("--train-samples-per-group", type=int, default=Config().train_samples_per_group)
    p_train.add_argument("--val-ratio", type=float, default=Config().val_ratio)
    p_train.add_argument("--infer-stride-sec", type=int, default=Config().infer_stride_sec)
    p_train.add_argument("--smooth-window-sec", type=int, default=Config().smooth_window_sec)
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

    return parser.parse_args()


def ensure_index(data_root: str, index_path: str) -> None:
    if not Path(index_path).exists():
        stats = build_index(data_root, index_path)
        logger.info("Index built: sessions=%s images=%s skipped=%s", stats.sessions, stats.images, stats.skipped)


def load_model(checkpoint_path: Path, cfg: Config, device: torch.device) -> SiameseResNetGRU:
    model = SiameseResNetGRU(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        pretrained=False,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        bidirectional=cfg.bidirectional,
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
    only_fold: int | None = None,
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
    fields = ["fold", "accuracy", "macro_f1", "class0_precision", "class0_recall", "class0_f1"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})


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

        cfg = Config(
            data_root=args.data_root,
            index_path=args.index_path,
            output_dir=output_dir,
            num_classes=args.num_classes,
            clip_sec=args.clip_sec,
            seq_len=args.seq_len,
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


if __name__ == "__main__":
    main()
