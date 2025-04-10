from bs4 import BeautifulSoup
import requests
from pathlib import Path
import os


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
    )
    return session


def get_content(session: requests.Session, url: str):
    request = session.get(url)
    html_doc = request.content
    return html_doc


def get_links(html_doc) -> list[str]:
    soup = BeautifulSoup(html_doc, "html.parser")
    results = []
    for tag in soup.article.find("ul", class_="stream-items").find_all(
        "li", class_="story-item", recursive=False
    ):
        results.append(tag.section.a["href"])
    return results


def scrape_article(article_html) -> str:
    soup = BeautifulSoup(article_html, "html.parser")
    result = []
    for tag in soup.article.find("div", class_="body").find_all("p"):
        for string in tag.stripped_strings:
            result.append(string)
    return " ".join(result)


def main():
    URL = "LINKS"
    session = create_session()
    links_content = get_content(session=session, url=URL)
    links = get_links(links_content)

    os.makedirs(name="./backend/scraper/articles", exist_ok=True)
    articles_path = Path("./backend/scraper/articles")

    for link in links:
        filename = link.rsplit("/", 1)[-1].replace(".html", ".txt")
        with open(articles_path / filename, "wt") as f:
            article_content = get_content(session=session, url=link)
            article_text = scrape_article(article_html=article_content)
            f.write(article_text)


if __name__ == "__main__":
    main()
