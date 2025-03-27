import torch
from torch.utils.data import DataLoader


def train_step(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accuracy_metric,
    precision_metric,
    device: torch.device,
):
    accuracy_metric.reset()
    precision_metric.reset()
    running_loss = 0.0
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        preds = torch.argmax(output.logits, dim=1)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy_metric(preds=preds, target=batch["labels"])
        precision_metric(preds=preds, target=batch["labels"])
        running_loss += output.loss.item()
    train_accuracy = accuracy_metric.compute().item()
    train_precision = precision_metric.compute().item()
    avg_loss = running_loss / len(train_dataloader)
    return {
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_loss": avg_loss,
    }


def val_step(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    accuracy_metric,
    precision_metric,
    device: torch.device,
):
    accuracy_metric.reset()
    precision_metric.reset()
    running_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            preds = torch.argmax(output.logits, dim=1)

            accuracy_metric(preds=preds, target=batch["labels"])
            precision_metric(preds=preds, target=batch["labels"])
            running_loss += output.loss.item()
        val_accuracy = accuracy_metric.compute().item()
        val_precision = precision_metric.compute().item()
        avg_loss = running_loss / len(eval_dataloader)
    return {
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_loss": avg_loss,
    }


def test_step(
    model: torch.nn.Module, test_dataloader: DataLoader, device: torch.device
):
    pass
