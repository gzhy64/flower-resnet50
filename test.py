from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from datasets import FlowerDataset, split_indices
from engine import evaluate
from models import ResNet50


"""
    用法：python test.py --checkpoint runs/20261021_172653
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估指定 ResNet50 模型的测试集表现"
    )

    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点文件路径")

    return parser.parse_args()


def main():
    args = parse_args()

    project_dir = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint.resolve() / "best.pt"

    print("使用的模型文件：", checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FlowerDataset(project_dir / "flower_photos", transform=transform)
    if dataset.class_to_idx != checkpoint["class_to_idx"]:
        raise ValueError("当前数据集的类别映射与模型检查点不一致")

    _, _, test_indices = split_indices(
        dataset.samples,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    class_names = checkpoint["class_names"]

    model = ResNet50(num_classes=len(class_names))

    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model = model.to(device)

    print("参数加载结果：", load_result)
    print("使用的模型轮次：", checkpoint["epoch"])
    print(f"保存的验证准确率：{checkpoint['val_acc']:.2%}")
    print("测试集图片数：", len(test_dataset))
    print("测试集批次数：", len(test_loader))

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"测试集平均损失：{test_loss:.4f}")
    print(f"测试集准确率：{test_acc:.2%}")


if __name__ == '__main__':
    main()