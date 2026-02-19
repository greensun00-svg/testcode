import asyncio
from playwright.async_api import async_playwright

async def main():
    print("Starting Playwright...")
    try:
        async with async_playwright() as p:
            print("Launching browser...")
            browser = await p.chromium.launch(headless=False)
            print("Browser launched. Creating page...")
            page = await browser.new_page()
            print("Page created. Navigating...")
            await page.goto("https://search.shopping.naver.com/catalog/42594163618")
            print("Navigated. Title:", await page.title())
            await browser.close()
            print("Browser closed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
