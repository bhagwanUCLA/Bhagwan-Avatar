import asyncio
import os
import logging
from scraper import PortfolioScraper, ScraperCache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scrapers():
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    yt_key = os.environ.get("YOUTUBE_API_KEY", "")
    
    if not gemini_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # Initialize scraper
    cache = ScraperCache("./test_cache")
    scraper = PortfolioScraper(
        gemini_api_key=gemini_key,
        youtube_api_key=yt_key,
        cache=cache,
        max_crawl_pages=2
    )

    # 1. Test YouTube Video (Transcript vs Gemini)
    print("\n--- Testing YouTube Video Summary ---")
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Never gonna give you up (likely has transcripts)
    docs = scraper._summarise_video(video_url, "test")
    for d in docs:
        print(f"Title: {d.title}")
        print(f"Content length: {len(d.content)}")
        print(f"Doc Type: {d.doc_type}")

    # 2. Test Playwright Crawl (Async)
    print("\n--- Testing Playwright Crawl ---")
    site_url = "https://example.com"
    try:
        # Note: scrape_portfolio now uses asyncio.run internally, 
        # so we call it directly (it's sync)
        docs = scraper.scrape_portfolio(site_url)
        for d in docs:
            print(f"Title: {d.title}")
            print(f"Section: {d.section}")
            if d.extra.get("is_cleaned_profile"):
                print("Found Cleaned Profile Doc!")
    except Exception as e:
        print(f"Crawl failed: {e}")
        print("Note: Ensure you have run 'playwright install chromium'")

if __name__ == "__main__":
    asyncio.run(test_scrapers())
