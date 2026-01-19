# VAS Time-Series (Rebuilt)

This project rebuilds the VAS time-series pipeline from scratch with a focus on speed and clean data flow.

## Core idea
- **Siamese model**: anchor = Normal window, target = VAS window.
- **VAS=0 is a special class**.
- **Other VAS values are split into uniform bins** over `(0, 100]`.

Example with `--num-classes 3`:
- class0: VAS = 0
- class1: VAS = 1..50
- class2: VAS = 51..100

## Data
This code reads already-extracted images at **3 fps** from:

```
/home/user/alcohol_exp/workspace/vas_detection/face_data_vas
```

Filename format (from extraction):
```
subj102_normal_t73s_f1100.png
subj102_vas20min_t919s_f13793_vas2_class0.png
```

The indexer parses:
- subject id
- session id (folder name)
- time (sec)
- VAS value (if present)

## Quickstart

Build index:

```bash
python main.py index \
  --data-root /home/user/alcohol_exp/workspace/vas_detection/face_data_vas \
  --index-path data/index.jsonl
```

Train (K-fold, default 9):

```bash
python main.py train \
  --num-classes 3 \
  --n-folds 9 \
  --batch-size 64 \
  --epochs 50
```

Outputs:
```
outputs/run_YYYYmmdd_HHMMSS/
  config.json
  train.log
  cv_results.csv
  fold_0/
    model_best.pt
    metrics.json
    timeseries/
      <session>_timeseries.csv
      <session>_timeseries.png
```

Run visualization only:
```bash
python main.py visualize --output-dir outputs/run_YYYYmmdd_HHMMSS
```

Coarse inference every 10 seconds:
```bash
python main.py visualize --output-dir outputs/run_YYYYmmdd_HHMMSS --infer-stride-sec 10
```

Run visualization directly from videos (covers non-extracted ranges):
```bash
python main.py visualize-video \
  --output-dir outputs/run_YYYYmmdd_HHMMSS \
  --video-root /home/user/alcohol_exp/database \
  --infer-stride-sec 10
```

Summarize CV metrics (accuracy, macro F1, class0 precision/recall/F1):
```bash
python main.py summarize-f1 --output-dir outputs/run_YYYYmmdd_HHMMSS
```

Summarize all runs under the outputs root:
```bash
python main.py summarize-f1 --outputs-root outputs
```

Write a summary CSV:
```bash
python main.py summarize-f1 --outputs-root outputs --csv-out outputs/summary.csv
```

Small hyperparameter sweep (quick validation):
```bash
python main.py sweep \
  --output-root outputs \
  --n-folds 3 \
  --epochs 10 \
  --lrs 1e-4,3e-4 \
  --batch-sizes 32,64 \
  --focal-gammas 2.0
```

Optuna tuning (small validation):
```bash
/home/user/alcohol_exp/workspace/vas_detection/.venv/bin/python scripts/optuna_tune.py \
  --output-root outputs \
  --n-folds 3 \
  --epochs 10 \
  --n-trials 8 \
  --lr-min 1e-4 \
  --lr-max 3e-3 \
  --batch-sizes 32,64 \
  --focal-gammas 1.0,2.0 \
  --flip-probs 0.0,0.5 \
  --brightness-jitters 0.0,0.1 \
  --contrast-jitters 0.0,0.1 \
  --noise-stds 0.0,0.02
```
Trial summaries are written to `outputs/optuna_trials_*.csv` by default (override with `--trial-csv`).

## Notes
- Time-series visualization uses **sliding 5s windows** with configurable stride.
- Moving average is applied to probabilities for smoother plots.
- VAS=0 precision/recall/F1 are included in `cv_results.csv`.
- Siamese anchor uses **0-5 min**; VAS=0 targets use **5-10 min** normal window.
- Visualization uses all available frames in the extracted dataset. Full-time coverage depends on what was extracted.
