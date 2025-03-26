def adjust_labels(sample):
    label = sample["label"]
    if label == 0:
        sample["label"] = 1
    elif label == 1:
        sample["label"] = 0
    else:
        sample["label"] = label
    return sample
