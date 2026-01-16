"""Entry point for the rebuilt VAS time-series pipeline."""

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch

from vas.config import Config
from vas.indexer import build_index
from vas.dataset import load_sessions
from vas.models import SiameseResNetGRU
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
    p_train.add_argument("--seq-len", type=int, default=Config().seq_len)
    p_train.add_argument("--n-folds", type=int, default=Config().n_folds)
    p_train.add_argument("--batch-size", type=int, default=Config().batch_size)
    p_train.add_argument("--epochs", type=int, default=Config().epochs)
    p_train.add_argument("--lr", type=float, default=Config().lr)
    p_train.add_argument("--backbone", default=Config().backbone)
    p_train.add_argument("--pretrained", action="store_true")
    p_train.add_argument("--no-amp", action="store_true")
    p_train.add_argument("--num-workers", type=int, default=Config().num_workers)
    p_train.add_argument("--train-samples-per-group", type=int, default=Config().train_samples_per_group)
    p_train.add_argument("--val-ratio", type=float, default=Config().val_ratio)
    p_train.add_argument("--infer-stride-sec", type=int, default=Config().infer_stride_sec)
    p_train.add_argument("--smooth-window-sec", type=int, default=Config().smooth_window_sec)
    p_train.add_argument("--gpus", default=Config().gpus)
    p_train.add_argument("--skip-visualize", action="store_true")

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


def run_visualizations(cfg: Config, output_dir: str, splits: List[Tuple[List[str], List[str], List[str]]]) -> None:
    sessions = load_sessions(cfg.index_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (_, _, test_ids) in enumerate(splits):
        fold_dir = Path(output_dir) / f"fold_{fold_idx}"
        model_path = fold_dir / "model_best.pt"
        if not model_path.exists():
            logger.warning("Missing model for fold %s: %s", fold_idx + 1, model_path)
            continue

        model = load_model(model_path, cfg, device)
        out_dir = fold_dir / "timeseries"
        out_dir.mkdir(parents=True, exist_ok=True)

        for session_id in test_ids:
            session = sessions.get(session_id)
            if session is None:
                continue
            try:
                data = predict_timeseries(model, session, cfg, device)
                save_timeseries_plot(out_dir, session, data, cfg)
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
            pretrained=args.pretrained,
            use_amp=not args.no_amp,
            num_workers=args.num_workers,
            train_samples_per_group=args.train_samples_per_group,
            val_ratio=args.val_ratio,
            infer_stride_sec=args.infer_stride_sec,
            smooth_window_sec=args.smooth_window_sec,
            gpus=args.gpus,
        )

        ensure_index(cfg.data_root, cfg.index_path)
        save_config(cfg, output_dir)

        results, splits = run_kfold(cfg, output_dir)
        write_cv_results(results, output_dir)

        if not args.skip_visualize:
            run_visualizations(cfg, output_dir, splits)


if __name__ == "__main__":
    main()
