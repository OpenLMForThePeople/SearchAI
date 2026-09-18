from playwright.sync_api import sync_playwright
from urllib.parse import quote


class GoogleBlockedError(Exception):
    """Google blocked the automated search request."""
    pass


def search_google(
    query: str,
    max_results: int = 10
) -> list[dict]:

    search_url = (
        "https://www.google.com/search?q="
        + quote(query)
    )

    results = []

    try:
        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )

            page = browser.new_page(
                locale="en-US",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/140.0.0.0 Safari/537.36"
                )
            )

            page.goto(
                search_url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            page.wait_for_timeout(2000)

            print(
                f"DEBUG: Page URL = {page.url}"
            )

            print(
                f"DEBUG: Page title = {page.title()}"
            )

            # Google blocked the request.
            if "/sorry/" in page.url:
                raise GoogleBlockedError(
                    "Google blocked the automated search request."
                )

            # Look for CAPTCHA / consent pages.
            body_text = page.locator(
                "body"
            ).inner_text()

            lower_text = body_text.lower()

            if (
                "captcha" in lower_text
                or "unusual traffic" in lower_text
            ):
                raise GoogleBlockedError(
                    "Google returned a CAPTCHA or "
                    "unusual-traffic page."
                )

            if "consent" in lower_text:
                raise GoogleBlockedError(
                    "Google returned a consent page."
                )

            # Look for Google result headings.
            h3s = page.locator("h3")

            print(
                f"DEBUG: h3 count = {h3s.count()}"
            )

            for i in range(
                min(h3s.count(), max_results)
            ):

                h3 = h3s.nth(i)

                title = h3.inner_text().strip()

                link = h3.locator(
                    "xpath=ancestor::a[1]"
                )

                if link.count() == 0:
                    continue

                result_url = link.get_attribute(
                    "href"
                )

                if not result_url:
                    continue

                if not result_url.startswith("http"):
                    continue

                results.append({
                    "title": title,
                    "channel": "",
                    "url": result_url,
                    "description": "",
                })

            browser.close()

    except GoogleBlockedError:
        # IMPORTANT:
        # Do not convert our custom error into [].
        raise

    except Exception as e:
        print(
            f"Google search failed: {e}"
        )
        return []

    print(
        f"DEBUG: Google returned "
        f"{len(results)} results"
    )

    return results