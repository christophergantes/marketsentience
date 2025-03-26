def main():
    import torch
    import mlflow
    import mlflow.transformers
    import torchmetrics
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    from datasets import load_dataset
    from data_setup import adjust_labels
    from utils import set_seeds, set_device
    from tqdm.auto import tqdm
    import os

    NUM_EPOCHS = 3
    LR = 0.00005
    BATCH_SIZE = 32
    SEED = 42

    device = set_device()
    print(f"[INFO] Torch device set to '{device}'")

    set_seeds(SEED)
    print(f"[INFO] Seeds set to {SEED}")

    checkpoint = "ProsusAI/finbert"
    dataset_name = "zeroshot/twitter-financial-news-sentiment"

    print(f"[INFO] Loading dataset '{dataset_name}'")
    raw_dataset = load_dataset(path=dataset_name)
    raw_dataset = raw_dataset.map(adjust_labels)

    def tokenize_function(sample):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return tokenizer(sample["text"], truncation=True)

    print("[INFO] Tokenizing dataset")
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)

    print("[INFO] Creating dataloaders")
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator
    )

    print(f"[INFO] Loading '{checkpoint}' model ")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)
    precision_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(
        device
    )

    mlflow.set_tracking_uri('http://localhost:5000')
    with mlflow.start_run():
        params = {
            "LEARNING_RATE": LR,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": NUM_EPOCHS,
            "SEED": SEED,
        }
        mlflow.log_params(params)
        for epoch in tqdm(range(NUM_EPOCHS)):
            accuracy_metric.reset()
            precision_metric.reset()
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
                mlflow.log_metric(
                    "train_loss_per_batch",
                    loss.item(),
                    step=epoch * len(train_dataloader) + step,
                )

            train_accuracy = accuracy_metric.compute().item()
            train_precision = precision_metric.compute().item()
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("train_precision", train_precision, step=epoch)

            accuracy_metric.reset()
            precision_metric.reset()
            model.eval()
            with torch.inference_mode():
                for step, batch in enumerate(eval_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    output = model(**batch)
                    preds = torch.argmax(output.logits, dim=1)

                    accuracy_metric(preds=preds, target=batch["labels"])
                    precision_metric(preds=preds, target=batch["labels"])

                val_accuracy = accuracy_metric.compute().item()
                val_precision = precision_metric.compute().item()
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                mlflow.log_metric("val_precision", val_precision, step=epoch)
        model.save_pretrained(f"./models/finbert")
        tokenizer.save_pretrained(f"./models/finbert")


if __name__ == "__main__":
    main()
