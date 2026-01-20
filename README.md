# VAS 時系列（再構築版）

本プロジェクトは VAS 時系列パイプラインを、速度とデータフローの明確さを重視して再構築したものです。

## 基本方針
- **Siamese モデル**: anchor = Normal 窓、target = VAS 窓。
- **VAS=0 を特別クラス**として扱う。
- **その他の VAS 値は (0, 100] を等間隔 bin に分割**。

`--num-classes 3` の例:
- class0: VAS = 0
- class1: VAS = 1..50
- class2: VAS = 51..100

## データ
本コードは、事前抽出済みの顔画像（**3 fps**）を読み込みます:

```
/home/user/alcohol_exp/workspace/vas_detection/face_data_vas
```

ファイル名フォーマット（抽出時）:
```
subj102_normal_t73s_f1100.png
subj102_vas20min_t919s_f13793_vas2_class0.png
```

インデクサが解析する情報:
- 被験者ID
- セッションID（フォルダ名）
- 時刻（秒）
- VAS 値（存在する場合）

## クイックスタート

インデックス作成:

```bash
python main.py index \
  --data-root /home/user/alcohol_exp/workspace/vas_detection/face_data_vas \
  --index-path data/index.jsonl
```

学習（K-fold、デフォルト 9）:

```bash
python main.py train \
  --num-classes 3 \
  --n-folds 9 \
  --batch-size 64 \
  --epochs 50
```

出力:
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

可視化のみ実行:
```bash
python main.py visualize --output-dir outputs/run_YYYYmmdd_HHMMSS
```

10秒間隔の粗い推論:
```bash
python main.py visualize --output-dir outputs/run_YYYYmmdd_HHMMSS --infer-stride-sec 10
```

動画から直接可視化（未抽出区間もカバー）:
```bash
python main.py visualize-video \
  --output-dir outputs/run_YYYYmmdd_HHMMSS \
  --video-root /home/user/alcohol_exp/database \
  --infer-stride-sec 10
```

CV 指標の要約（accuracy / macro F1 / class0 precision/recall/F1）:
```bash
python main.py summarize-f1 --output-dir outputs/run_YYYYmmdd_HHMMSS
```

outputs 直下の全 run を要約:
```bash
python main.py summarize-f1 --outputs-root outputs
```

要約 CSV を出力:
```bash
python main.py summarize-f1 --outputs-root outputs --csv-out outputs/summary.csv
```

小規模ハイパラスイープ（簡易検証）:
```bash
python main.py sweep \
  --output-root outputs \
  --n-folds 3 \
  --epochs 10 \
  --lrs 1e-4,3e-4 \
  --batch-sizes 32,64 \
  --focal-gammas 2.0
```

Optuna チューニング（小規模検証）:
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
試行サマリは既定で `outputs/optuna_trials_*.csv` に出力されます（`--trial-csv` で変更可）。
CUDA の pin-memory エラーが出る場合は `--no-pin-memory`（必要なら `--num-workers 0`）を追加してください。

## 注意点
- 時系列可視化は **5秒窓スライド**で、stride は調整可能です。
- スムージング（移動平均）で確率曲線を平滑化します。
- VAS=0 の precision/recall/F1 は `cv_results.csv` に含まれます。
- Siamese の anchor は **0-5分**、VAS=0 の target は **5-10分**の normal 窓を使用します。
- 可視化は抽出済みフレームに依存するため、全時間を必ずしもカバーしません。
