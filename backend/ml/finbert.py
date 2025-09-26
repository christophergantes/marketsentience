from transformers import AutoTokenizer, AutoModelForSequenceClassification

CHECKPOINT = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
model.eval()
