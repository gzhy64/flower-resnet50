from pathlib import Path
import time
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataclasses import asdict

from datasets import FlowerDataset, split_indices
from models import ResNet50
from engine import train_one_epoch, evaluate
from config import TrainConfig
from experiment import create_experiment, append_metrics


def parse_args():
    defaults = TrainConfig()

    parser = argparse.ArgumentParser(
        description="训练 ResNet50 花卉分类模型"
    )

    parser.add_argument("--epochs",type=int,default=defaults.num_epochs, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="每个批次的图片数量")
    parser.add_argument("--lr", type=float, default=defaults.learning_rate, help="学习率")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers, help="数据加载使用的工作进程数量")

    return parser.parse_args()

def main():
    args = parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )

    project_dir = Path(__file__).resolve().parent
    print("本次训练配置：", config)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FlowerDataset(
        project_dir / "flower_photos",
        transform=transform,
    )
    # 每一张图片的索引
    train_indices, val_indices, test_indices = split_indices(dataset.samples)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet50(num_classes=len(dataset.class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
    )

    num_epochs = config.num_epochs
    best_val_acc = -1.0
    best_epoch = 0

    # 设置保存实验数据路径
    run_dir = create_experiment(project_dir, config)
    checkpoint_path = run_dir / "best.pt"
    print(f"本次实验目录：{run_dir}")

    start_time = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        print(f"\n开始第 {epoch}/{num_epochs} 轮")

        train_loss, train_acc = train_one_epoch(
            model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = evaluate(
            model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"第 {epoch}/{num_epochs} 轮 | "
            f"训练损失 {train_loss:.4f}，准确率 {train_acc:.2%} | "
            f"验证损失 {val_loss:.4f}，准确率 {val_acc:.2%}"
        )

        append_metrics(
            run_dir,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # 保存检查点
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": dataset.class_names,
                "class_to_idx": dataset.class_to_idx,
                "train_config": asdict(config),
            }

            torch.save(checkpoint, checkpoint_path)

            print(
                f"已保存最佳模型：第 {epoch} 轮，"
                f"验证准确率 {val_acc:.2%}"
            )

    elapsed = time.perf_counter() - start_time
    print(f"{num_epochs} 轮总耗时：{elapsed:.2f} 秒")


    print(
        f"\n最佳验证准确率：{best_val_acc:.2%}，"
        f"来自第 {best_epoch} 轮"
    )
    print(f"模型保存位置：{checkpoint_path}")



if __name__ == '__main__':
    main()