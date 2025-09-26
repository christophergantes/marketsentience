from fastapi import FastAPI
from app.routers import sentiment

app = FastAPI()


@app.get("/")
def main():
    return {"hello": "world"}


app.include_router(sentiment.router)
