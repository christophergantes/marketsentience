def main():
    import torch
    import mlflow
    import torchmetrics
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    from datasets import load_dataset
    from engine import train_step, val_step
    from data_setup import adjust_labels, tokenize_dataset, create_dataloaders
    from utils import set_seeds, set_device
    from tqdm.auto import tqdm

    NUM_EPOCHS = 10
    SEED = 42
    lr_params = [0.000001]
    batch_size_params = [32]

    checkpoint = "ProsusAI/finbert"
    dataset_name = "zeroshot/twitter-financial-news-sentiment"

    device = set_device()
    print(f"[INFO] Torch device set to '{device}'")

    set_seeds(SEED)
    print(f"[INFO] Seeds set to {SEED}")

    print(f"[INFO] Loading dataset '{dataset_name}'")
    raw_dataset = load_dataset(path=dataset_name)
    raw_dataset = raw_dataset.map(adjust_labels)

    print("[INFO] Tokenizing dataset")
    tokenized_dataset = tokenize_dataset(raw_dataset, checkpoint)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    mlflow.set_experiment("Hyperparameter_tuning")
    for batch_size in batch_size_params:
        print("[INFO] Creating dataloaders")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        data_collator = DataCollatorWithPadding(tokenizer)
        train_dataloader, val_dataloader = create_dataloaders(
            tokenized_dataset["train"],
            tokenized_dataset["validation"],
            batch_size,
            collate_fn=data_collator,
        )

        for lr in lr_params:
            print(f"[INFO] Loading '{checkpoint}' model")
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(
                device
            )
            model_name = checkpoint.split("/")[-1]
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

            mlflow.set_tracking_uri("http://localhost:8080")
            with mlflow.start_run():
                params = {
                    "MODEL": model_name,
                    "LEARNING_RATE": lr,
                    "BATCH_SIZE": batch_size,
                    "EPOCHS": NUM_EPOCHS,
                    "SEED": SEED,
                }
                mlflow.log_params(params)

                accuracy_metric = torchmetrics.Accuracy(
                    task="multiclass", num_classes=3
                ).to(device)
                precision_metric = torchmetrics.Precision(
                    task="multiclass", num_classes=3
                ).to(device)

                for epoch in tqdm(range(NUM_EPOCHS)):
                    train_results = train_step(
                        model,
                        train_dataloader,
                        optimizer,
                        accuracy_metric,
                        precision_metric,
                        device,
                    )
                    val_results = val_step(
                        model, val_dataloader, accuracy_metric, precision_metric, device
                    )

                    mlflow.log_metrics(train_results, step=epoch)
                    mlflow.log_metrics(val_results, step=epoch)
                    model_save_name = (
                        f"{model_name}_batch_{batch_size}_lr_{lr}_epoch_{epoch}"
                    )
                    model.save_pretrained(f"./backend/ml/models/{model_save_name}")


if __name__ == "__main__":
    main()
