from bs4 import BeautifulSoup
import requests


# URL = "https://finance.yahoo.com/news/live/stock-market-today-dow-sp-500-nasdaq-futures-rise-after-chaotic-day-on-wall-street-220829617.html"
# URL = "https://finance.yahoo.com/news/off-price-and-discount-retailers-poised-to-outperform-as-trump-tariffs-continue-to-batter-markets-170956529.html"

def scrape_article(article_html):
    soup = BeautifulSoup(article_html, "html.parser")
    result = []
    for tag in soup.article.find("div", class_="atoms-wrapper").find_all("p", recursive=False):
        for string in tag.stripped_strings:
            result.append(string)
    print(" ".join(result))
    
def get_links(quote_news_url):
    session = requests.Session()
    session.headers.update( {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    } )
    request = session.get(URL)
    


def main():
  pass
    

if __name__ == "__main__":
    main()
