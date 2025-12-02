from fastapi import FastAPI

from api import sentiment

app = FastAPI()


@app.get("/")
def main():
    return {"hello": "world"}


app.include_router(sentiment.router)
