from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 0.01
    num_epochs: int = 20


if __name__ == '__main__':
    config = TrainConfig()

    print(config)
    # 转换成字典格式
    print(asdict(config))