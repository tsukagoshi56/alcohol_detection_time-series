# VAS 静止画モデル (ResNet)

本プロジェクトは、顔画像からVAS（主観的アルコール感）を推定するモデルです。
以前の時系列モデル (CNN+RNN) を廃止し、**単一フレームベースの静止画モデル (Siamese ResNet)** に変更しました。

## 基本方針
- **Siamese モデル**: 
  - Anchor: 最初の5分間 (0-5分) の Normal 状態からランダムに選ばれたフレーム。
  - Target: 推定対象のフレーム。
  - 入力: 2枚の画像を ResNet Backbone に通し、特徴量の差分を分類器に入力。
- **クラス分類**:
  - class0: VAS = 0 (Normal)
  - class1..: VAS > 0 を等間隔に分割 (例: 1-50, 51-100)

## 環境構築 (uv)

本プロジェクトはパッケージマネージャーとして `uv` を使用します。

### インストール
まだ `uv` がない場合:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 依存関係の同期
プロジェクトルートで以下を実行して環境を構築します:
```bash
uv sync
```

## データ
事前抽出済みの顔画像（png/jpg）を使用します。
デフォルトパス: `../vas_detection/face_data_vas` (環境に合わせて変更してください)

ファイル名フォーマット例:
```
subj101_normal_t73s_f1100.png
subj101_vas20min_t919s_f13793_vas2_class0.png
```

## クイックスタート

すべてのコマンドは `uv run` を介して実行します。

### 1. インデックス作成
画像ファイルのリストを作成します。

```bash
uv run main.py index \
  --data-root /path/to/face_data_vas \
  --index-path data/index.jsonl
```

### 2. 学習 (K-fold Cross Validation)
デフォルトは 9-fold です。

```bash
uv run main.py train \
  --data-root /path/to/face_data_vas \
  --num-classes 3 \
  --n-folds 9 \
  --batch-size 32 \
  --epochs 50 \
  --output-dir outputs/experiment_v1
```

**主なオプション:**
- `--clip-sec`: 推論単位の時間窓（秒）。デフォルト5秒。この期間から1フレームをサンプリングします。
- `--backbone`: `resnet18` (デフォルト) または `resnet34`。
- `--train-samples-per-group`: 1つのVAS区間から学習に使うサンプル数。

### 3. 可視化
学習済みモデルを使って推論結果をプロットします。

```bash
uv run main.py visualize \
  --output-dir outputs/experiment_v1 \
  --infer-stride-sec 10
```

### 4. 動画からの直接推論
抽出済み画像がない動画ファイルに対しても推論可能です (MediaPipeが必要)。

```bash
uv run main.py visualize-video \
  --output-dir outputs/viz_video \
  --video-root /path/to/videos \
  --infer-stride-sec 5
```

## 出力フォルダ構成
```
outputs/experiment_v1/
  config.json          # 学習設定
  train.log            # ログ
  cv_results.csv       # 全foldのスコア
  fold_0/              # foldごとのディレクトリ
    model_best.pt      # 最良モデル
    metrics.json       # 指標
    timeseries/        # 時系列プロット
      subj101_...timeseries.png
      subj101_...timeseries.csv
```

## ハイパーパラメータ探索
Config オブジェクト (`vas/config.py`) またはコマンドライン引数で調整可能です。
小規模な探索には `sweep` コマンドも利用できます。

```bash
uv run main.py sweep --n-folds 3 --epochs 10 --lrs 1e-4,1e-3
```
