from typing import List

from trafilatura import extract

import asyncio
from playwright.async_api import Playwright, Page, expect

class WebSearch:
    def __init__(self, num_results: int, term: str, username: str = "", password: str = "", debug: bool = False):
        """
        :param num_results: Number of results to return
        :param term: Search term
        :param username: Username for login
        :param password: Password for login
        :param debug: Whether to run in debug mode
        """
        self.num_results = num_results
        self.term = term
        self.username = username
        self.password = password
        self.requires_login = True if username and password else False
        self.debug = debug

    async def extract_content(self, page: Page, href: str) -> str:
        """
        Extract content from a search result page
        :param page: Page object
        :param href: Link to the search result page
        """
        await page.goto(href, wait_until='networkidle')
        content = await page.content()
        return extract(content)

    async def run(self, playwright: Playwright) -> List[str]:
        """
        Main function
        :param playwright: Playwright object
        """
        browser = await playwright.webkit.launch(headless= not self.debug, slow_mo= 1000 if self.debug else 0)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36")
        page = await context.new_page()

        if self.requires_login:
            await page.goto('https://www.website.com/login', wait_until='networkidle')

            # Login
            await page.locator('#userName').fill(self.username)
            await page.locator('#password').fill(self.password)
            async with page.expect_navigation():
                await page.locator('#btnLogin').click()
            await expect(page.locator('#searchbox')).to_be_visible()
        else:
            await page.goto('https://www.website.com/search', wait_until='networkidle')

        # Search
        await page.locator('//*[@id="tbSearch"]').fill(self.term)
        async with page.expect_navigation():
            await page.get_by_role("button", name="Submit search").click()
        await page.wait_for_selector('#results-container')

        # Extracting links from the search results
        hrefs = await page.eval_on_selector_all('.result-list-item a', 'links => links.map(link => link.href)')
        hrefs = [href for href in hrefs if 'type=default&display_rank=' in href]

        # Extract content from the first n links
        tasks = []
        for href in hrefs[:self.num_results]:
            search_page = await context.new_page()
            tasks.append(asyncio.ensure_future(self.extract_content(search_page, href)))
        docs = await asyncio.gather(*tasks)

        # Gather docs and links
        results = list(zip(hrefs, docs))

        # Clean up
        await context.close()
        await browser.close()

        return results
