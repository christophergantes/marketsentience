import torch
from services.finbert import model, tokenizer
from services.massive_client import get_latest_news
from fastapi import APIRouter, Query

router = APIRouter()

labels = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}


@router.get("/sentiment")
def get_sentiment(ticker: str = Query(..., min_length=1, max_length=5), limit: int = 1):
    articles = get_latest_news(ticker=ticker, limit=limit)
    headlines = [a.title for a in articles]

    tokens = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**tokens)

    preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
    sentiments = [labels[p] for p in preds]

    return {"ticker": ticker, "results": list(zip(headlines, sentiments))}
