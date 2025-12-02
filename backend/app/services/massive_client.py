from dotenv import load_dotenv
from massive import RESTClient
from massive.exceptions import BadResponse

from core.config import settings

load_dotenv()

MASSIVE_API_KEY = settings.MASSIVE_API_KEY

if not MASSIVE_API_KEY:
    raise ValueError("MASSIVE_API_KEY is not set in environment variables")

client = RESTClient(api_key=MASSIVE_API_KEY, pagination=False)


def get_latest_news(ticker: str | None = None, limit: int = 1):
    try:
        return [
            n
            for n in client.list_ticker_news(
                ticker=ticker, order="desc", limit=limit, sort="published_utc"
            )
        ]
    except BadResponse:
        return []
