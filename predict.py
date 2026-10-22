from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms

from models import ResNet50


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用指定 ResNet50 模型预测单张图片"
    )

    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点文件路径")
    parser.add_argument("--image", type=Path, required=True, help="待预测图片路径")

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve() / "best.pt"
    image_path = args.image.resolve()

    print("使用的模型文件：", checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    class_names = checkpoint["class_names"]

    model = ResNet50(num_classes=len(class_names))

    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.to(device)
    model.eval()

    print("参数加载结果：", load_result)
    print("模型保存轮次：", checkpoint["epoch"])
    print(f"保存的验证准确率：{checkpoint['val_acc']:.2%}")
    print("类别顺序：", class_names)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with Image.open(image_path) as pil_img:
        image = pil_img.convert("RGB")

    image_tensor = transform(image)

    # 添加批次保持和loader相同维度
    images = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

    pred_index = probabilities.argmax(dim=1).item()
    pred_name = class_names[pred_index]
    pred_probability = probabilities[0, pred_index].item()

    print("输入形状：", images.shape)
    print("输出形状：", logits.shape)
    print("图片路径：", image_path)
    print("预测类别编号：", pred_index)
    print("预测类别名称：", pred_name)
    print(f"预测类别概率：{pred_probability:.2%}")




if __name__ == '__main__':
    main()
