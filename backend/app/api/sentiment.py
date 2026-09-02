import torch
from services.finbert import model, tokenizer
from services.massive_client import get_latest_news
from fastapi import APIRouter, Request

router = APIRouter()

labels = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}


@router.get("/sentiment/{ticker}")
def get_sentiment(ticker: str, limit: int = 1, request: Request = None):

    print("Request Headers:")
    for k, v in request.headers.items():
        print(f"{k}: {v}")

    # Log query parameters
    print("Query params:", dict(request.query_params))

    # You can also log cookies
    print("Cookies:", request.cookies)


    articles = get_latest_news(ticker=ticker, limit=limit)
    if not articles:
        return {"ticker": ticker, "results": []}
    headlines = [a.title for a in articles]

    tokens = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**tokens)

    preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
    sentiments = [labels[p] for p in preds]

    return {"ticker": ticker, "results": list(zip(headlines, sentiments))}
