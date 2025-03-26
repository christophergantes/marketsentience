import torch
from torch.utils.data import DataLoader
import mlflow
import torchmetrics

accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
precision_metric = torchmetrics.Precision(task="multiclass", num_classes=3)


def train_step(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.to(device)
    accuracy_metric.to(device)
    precision_metric.to(device)

    train_loss = []
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        predictions = torch.argmax(output.logits, dim=1)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy_metric(preds=predictions, target=batch["labels"])
        precision_metric(preds=predictions, target=batch["labels"])
        train_loss.append(loss.item())

    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    return {"accuracy": accuracy, "precision": precision, "loss": train_loss}


def eval_step(
    model: torch.nn.Module, eval_dataloader: DataLoader, device: torch.device
):
    model.to(device)
    accuracy_metric.to(device)
    precision_metric.to(device)

    model.eval()
    eval_loss = []
    with torch.inference_mode():
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            predictions = torch.argmax(output.logits, dim=1)

            accuracy_metric(preds=predictions, target=batch["labels"])
            precision_metric(preds=predictions, target=batch["labels"])
            eval_loss.append(output.loss.item())
            mlflow.log_metric("eval_loss", output.loss.item(), step=step)
    accuracy = accuracy_metric.compute().item()
    precision = accuracy_metric.compute().item()

    return {"accuracy": accuracy, "precision": precision, "loss": eval_loss}


def test_step(
    model: torch.nn.Module, test_dataloader: DataLoader, device: torch.device
):
    pass
