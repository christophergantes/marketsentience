import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    MASSIVE_API_KEY: str = os.environ.get("MASSIVE_API_KEY", "")
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./db.sqlite3")


settings = Settings()
