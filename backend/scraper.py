import requests
from bs4 import BeautifulSoup
import urllib.parse

class GoogleNewsScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_fallback_data(self, query):
        encoded_query = urllib.parse.quote(query)
        return [
            {
                "title": f"Search '{query[:30]}...' on Google News",
                "source": "Google News (Manual Verify)",
                "link": f"https://www.google.com/search?q={encoded_query}&tbm=nws",
                "snippet": "Live scraping was limited. Click here to verify this headline directly on Google News."
            }
        ]

    def search(self, query):
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}&tbm=nws"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # This selector might need adjustment based on Google's changing HTML
            # Currently targeting the standard news result container
            for item in soup.select('div.SoaBEf')[:5]:
                title_el = item.select_one('div.MBeuO')
                source_el = item.select_one('.NUnG9d span')
                link_el = item.select_one('a.WlydOe')
                snippet_el = item.select_one('.GI74Re')
                
                if title_el and link_el:
                    results.append({
                        "title": title_el.get_text(),
                        "source": source_el.get_text() if source_el else "Unknown",
                        "link": link_el['href'],
                        "snippet": snippet_el.get_text() if snippet_el else ""
                    })
            
            if not results:
                print("Scraping failed or blocked. Using fallback data.")
                return self.get_fallback_data(query)
            
            return results
        except Exception as e:
            print(f"Error scraping Google News: {e}")
            print("Using fallback data due to error.")
            return self.get_fallback_data(query)

if __name__ == "__main__":
    scraper = GoogleNewsScraper()
    print(scraper.search("Aliens landed in New York"))
