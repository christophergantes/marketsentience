from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def adjust_labels(sample):
    label = sample["label"]
    if label == 0:
        sample["label"] = 1
    elif label == 1:
        sample["label"] = 0
    else:
        sample["label"] = label
    return sample


def tokenize_dataset(dataset, checkpoint):
    def tokenize_function(sample):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return tokenizer(sample["text"], truncation=True)

    return dataset.map(tokenize_function, batched=True)


def create_dataloaders(train_dataset, val_dataset, batch_size, collate_fn=None):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader
