import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_history(history_path):
    history = []

    with history_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            metrics = {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_loss": float(row["val_loss"]),
                "val_acc": float(row["val_acc"]),
            }

            history.append(metrics)

    return history


def plot_history(history, output_path):
    epochs = [metrics["epoch"] for metrics in history]

    train_losses = [metrics["train_loss"] for metrics in history]
    val_losses = [metrics["val_loss"] for metrics in history]

    train_accs = [metrics["train_acc"] * 100 for metrics in history]
    val_accs = [metrics["val_acc"] * 100 for metrics in history]

    best_metrics = max(history, key=lambda metrics: metrics["val_acc"])
    best_epoch = best_metrics["epoch"]
    best_val_acc = best_metrics["val_acc"] * 100

    fig, axes = plt.subplots(1,2, figsize=(12, 4), layout="constrained")

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Cross-entropy loss")

    axes[1].plot(epochs, train_accs, label="Train")
    axes[1].plot(epochs, val_accs, label="Validation")
    axes[1].scatter(best_epoch, best_val_acc, color="red", s=60, zorder=3, label=f"Best val: {best_val_acc:.2f}% (epoch {best_epoch})")
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取并绘制训练记录"
    )

    parser.add_argument("--run-dir", type=Path, required=True, help="实验结果目录")

    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = args.run_dir.resolve()
    history_path = run_dir / "history.csv"

    history = read_history(history_path)

    output_path = run_dir / "training_curves.png"

    plot_history(history, output_path)

    print("训练曲线已保存：", output_path)
    print("图片文件已生成：", output_path.is_file())


if __name__ == '__main__':
    main()