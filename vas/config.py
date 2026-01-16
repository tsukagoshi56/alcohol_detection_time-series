"""Configuration defaults for the rebuilt VAS pipeline."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Config:
    # Data
    data_root: str = "/home/user/alcohol_exp/workspace/vas_detection/face_data_vas"
    index_path: str = "data/index.jsonl"
    output_dir: str = "outputs"

    # Labels
    num_classes: int = 3  # class 0 is VAS=0, class 1.. are uniform bins over (0, 100]

    # Sampling
    clip_sec: int = 5
    seq_len: int = 15  # 3 fps * 5 sec
    min_frames_per_window: int = 3

    # Training samples per VAS window
    train_samples_per_group: int = 6
    val_samples_per_group: int = 1
    test_samples_per_group: int = 1

    # CV
    n_folds: int = 9
    val_ratio: float = 0.15
    seed: int = 42

    # Model
    backbone: str = "resnet18"
    pretrained: bool = False
    rnn_hidden: int = 256
    rnn_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.5

    # Optim
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 10

    # Runtime
    use_amp: bool = True
    num_workers: int = 8
    gpus: Optional[str] = None  # e.g. "0,1,2,3"

    # Image
    img_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Time-series visualization
    infer_stride_sec: int = 1
    smooth_window_sec: int = 15
