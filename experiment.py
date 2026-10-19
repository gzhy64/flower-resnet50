import csv
import json
from dataclasses import asdict
from datetime import datetime


"""
    每一次训练记录实验数据
"""

METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
]


def create_experiment(project_dir, config):
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = project_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            asdict(config),
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 每一轮指标
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()

    return run_dir



def append_metrics(run_dir, metrics):
    history_path = run_dir / "history.csv"
    # 保存每一轮指标
    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow(metrics)

