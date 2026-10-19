import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    # 累计预测正确的图片数
    total_acc = 0
    total_samples = 0

    for batch_index, (images, labels) in  enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # 批次大小8，最后一次是6
        batch_size = images.shape[0]
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_acc += (predictions == labels).sum().item()
        total_samples += batch_size

        # if batch_index % 50 == 0 or batch_index == len(loader):
        #     print(
        #         f"训练进度：{batch_index}/{len(loader)} 批，"
        #         f"已处理 {total_samples} 张"
        #     )

    epoch_loss = total_loss / total_samples
    epoch_acc = total_acc / total_samples

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.shape[0]
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * batch_size
            total_acc += (predictions == labels).sum().item()
            total_samples += batch_size

            # if batch_index % 10 == 0 or batch_index == len(loader):
            #     print(
            #         f"评估进度：{batch_index}/{len(loader)} 批，"
            #         f"已处理 {total_samples} 张"
            #     )

    epoch_loss = total_loss / total_samples
    epoch_acc = total_acc / total_samples

    return epoch_loss, epoch_acc
