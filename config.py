from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 100
    seed: int = 2026

    momentum: float = 0.9
    weight_decay: float = 5e-4
    min_lr:float = 1e-7

    label_smoothing: float = 0.1



if __name__ == '__main__':
    config = TrainConfig()

    print(config)
    # 转换成字典格式
    print(asdict(config))