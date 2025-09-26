from dotenv import load_dotenv
from polygon import RESTClient
from polygon.rest.models import TickerNews
import os

load_dotenv()
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
client = RESTClient(api_key=POLYGON_API_KEY, pagination=False)


def get_latest_news(ticker: str | None = None, limit: int = 1) -> list[TickerNews]:
    news = []
    for n in client.list_ticker_news(
        ticker=ticker, order="desc", limit=limit, sort="published_utc"
    ):
        news.append(n)

    return news
