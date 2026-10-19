from pathlib import Path

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import random


# 找到类别名，以及对应的类别号
def find_classes(data_dir):
    root = Path(data_dir)

    class_names = []
    for entry in root.iterdir():
        if entry.is_dir():
            class_names.append(entry.name)

    class_names.sort()

    class_to_idx = {}
    for class_index, class_name in enumerate(class_names):
        class_to_idx[class_name] = class_index

    return class_names, class_to_idx


# 每个图片对应的类别
def collect_samples(data_dir, class_to_idx):
    root = Path(data_dir)

    samples = []
    image_extensions = {".jpg", ".jpeg", ".png"}
    for class_name, class_index in class_to_idx.items():
        class_dir = root / class_name

        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                samples.append((image_path, class_index))

    return samples


def split_indices(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)

    indices_by_class = {}

    for index, sample in enumerate(samples):
        label = sample[1]

        if label not in indices_by_class:
            indices_by_class[label] = []

        indices_by_class[label].append(index)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_indices in indices_by_class.values():
        rng.shuffle(class_indices)

        num_train = int(len(class_indices) * train_ratio)
        num_val = int(len(class_indices) * val_ratio)

        train_indices.extend(class_indices[:num_train])

        val_indices.extend(class_indices[num_train:num_train + num_val])

        test_indices.extend(class_indices[num_train + num_val:])

    return train_indices, val_indices, test_indices


class FlowerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.class_names, self.class_to_idx = find_classes(data_dir)
        self.samples = collect_samples(data_dir, self.class_to_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        with Image.open(img_path) as pil_img:
            img = pil_img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def main():
    project_dir = Path(__file__).resolve().parent

    # 调整图片尺寸，转成张量
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 数据集
    dataset = FlowerDataset(project_dir / "flower_photos", transform=transform)

    # 把多张图片组成一个批次
    loader = DataLoader(dataset,batch_size=8, shuffle=True, num_workers=0)

    images, labels = next(iter(loader))

    print(images.shape)
    print(images.dtype)
    print(labels)
    print(labels.shape)
    print(labels.dtype)
    print(len(loader))


if __name__ == '__main__':
    main()
