"""Small Optuna-based hyperparameter search for the VAS model."""

from __future__ import annotations

import argparse
import statistics
from datetime import datetime
from pathlib import Path
from typing import List

import optuna

from vas.config import Config
from vas.indexer import build_index
from vas.trainer import run_kfold
from vas.utils import save_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for VAS pipeline")
    parser.add_argument("--data-root", default=Config().data_root)
    parser.add_argument("--index-path", default=Config().index_path)
    parser.add_argument("--output-root", default=Config().output_dir)
    parser.add_argument("--num-classes", type=int, default=Config().num_classes)
    parser.add_argument("--clip-sec", type=int, default=Config().clip_sec)
    parser.add_argument("--seq-len", type=int, default=Config().seq_len)
    parser.add_argument("--train-samples-per-group", type=int, default=Config().train_samples_per_group)
    parser.add_argument("--val-ratio", type=float, default=Config().val_ratio)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=Config().num_workers)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=Config().seed)
    parser.add_argument("--batch-sizes", default="32,64")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=3e-3)
    parser.add_argument("--focal-gammas", default="1.0,2.0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-focal", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--gpus", default=Config().gpus)
    parser.add_argument("--study-name", default="vas_optuna")
    parser.add_argument("--storage", default=None, help="Optuna storage (e.g. sqlite:///study.db)")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def parse_list(value: str, cast) -> List:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def ensure_index(data_root: str, index_path: str) -> None:
    if not Path(index_path).exists():
        build_index(data_root, index_path)


def main() -> None:
    args = parse_args()
    ensure_index(args.data_root, args.index_path)

    batch_sizes = parse_list(args.batch_sizes, int)
    if not batch_sizes:
        raise ValueError("No batch sizes provided.")

    focal_gammas = parse_list(args.focal_gammas, float)
    if args.no_focal:
        focal_gammas = [Config().focal_gamma]

    base_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ""
    if args.tag:
        tag = f"_{args.tag.strip().replace(' ', '_')}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_sizes)
        gamma = trial.suggest_categorical("focal_gamma", focal_gammas)

        run_name = f"run_{base_stamp}{tag}_optuna_t{trial.number:03d}"
        output_dir = Path(args.output_root) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(str(output_dir / "train.log"), force=True)

        cfg = Config(
            data_root=args.data_root,
            index_path=args.index_path,
            output_dir=str(output_dir),
            num_classes=args.num_classes,
            clip_sec=args.clip_sec,
            seq_len=args.seq_len,
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
            seed=args.seed,
        )

        save_config(cfg, str(output_dir))

        results, _ = run_kfold(cfg, str(output_dir))
        macro_f1 = [r.get("macro_f1", 0.0) for r in results]
        score = statistics.mean(macro_f1) if macro_f1 else 0.0
        trial.set_user_attr("output_dir", str(output_dir))
        return score

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print("Best trial:", best.number)
    print("Best value:", best.value)
    print("Params:", best.params)
    if "output_dir" in best.user_attrs:
        print("Output dir:", best.user_attrs["output_dir"])


if __name__ == "__main__":
    main()
