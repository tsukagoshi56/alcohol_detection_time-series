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

## Notes
- Time-series visualization uses **sliding 5s windows** with configurable stride.
- Moving average is applied to probabilities for smoother plots.
- VAS=0 precision/recall/F1 are included in `cv_results.csv`.
- Siamese anchor uses **0-5 min**; VAS=0 targets use **5-10 min** normal window.
