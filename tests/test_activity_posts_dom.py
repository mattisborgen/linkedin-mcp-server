"""Browser-DOM tests for the post programs in ``linkedin/activity.py``.

The unit suite mocks ``page.evaluate``, so the post-presence wait and the
mounted-post read never execute there; the tests in
``tests/linkedin/test_activity.py`` hand the reader records shaped like the
program's output. These cases run the programs in headless chromium, and run
the whole read once against a feed that loads its next batch on a real End key
press.

Every fixture drives a synthetic container, so each case is a claim about the
algorithm rather than about LinkedIn's markup. That includes the repost: an
avatar link to the original author ahead of the body, with the reposter named
only by a text link above it, is the shape the author read assumes and has not
been measured on a live feed. Skipped automatically when chromium is not
installed; run locally after ``uv run patchright install chromium --no-shell``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import json

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.linkedin import activity as activity_module
from linkedin_mcp_server.linkedin.activity import ActivityReader
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.session import PageSession

#: CI uses ``--dist loadgroup``. Keep every test that launches Chromium on one
#: worker so browser startups cannot compete with the DOM cases' wall-clock
#: timers.
#: Without that distribution mode the group mark is inert.
pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

FEED_URL = "https://www.linkedin.com/in/ada-lovelace/recent-activity/all/"
COMPANY_FEED_URL = (
    "https://www.linkedin.com/company/acme/posts/?feedView=all&viewAsMember=true"
)


@pytest.fixture
async def dom_page():
    """Real chromium page, or skip when no browser is installed.

    ``channel="chromium"`` names the browser this project installs. Without it
    Playwright picks the *binary* from the ``headless`` flag alone and asks for
    ``chromium-headless-shell``, which nothing here installs since the setup
    moved to ``--no-shell``.
    """
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chromium", headless=True)
            page = await browser.new_page()
        except Exception as exc:  # browser binary missing
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


async def serve(page: Page, url: str, html: str) -> None:
    """Load *html* at a LinkedIn *url*, which every read program requires."""
    await page.route(
        "https://www.linkedin.com/**",
        lambda route: route.fulfill(content_type="text/html", body=html),
    )
    await page.goto(url)


def urn(days_ago: float) -> str:
    """An activity URN whose id encodes a publish time *days_ago* before now."""
    published = datetime.now(UTC) - timedelta(days=days_ago)
    return f"urn:li:activity:{int(published.timestamp() * 1000) << 22 | 7}"


def actor(href: str, name: str) -> str:
    """The avatar link and the name link above a post's body."""
    return f'<a href="{href}"><img alt=""></a><a href="{href}"><span>{name}</span></a>'


def post(activity: str, author_href: str, author: str, body: str) -> str:
    return (
        f'<div data-urn="{activity}">{actor(author_href, author)}'
        f"<p>{body}</p><span>12 reactions</span></div>"
    )


def repost(activity: str, reposter_href: str, reposter: str, original: str) -> str:
    """Someone else's post under a line naming the reposter by a text link."""
    return (
        f'<div data-urn="{activity}">'
        f'<div><a href="{reposter_href}">{reposter}</a> reposted this</div>'
        f"{actor('/in/charles-babbage/', 'Charles Babbage')}"
        f"<p>{original}</p><span>480 reactions</span></div>"
    )


def feed(batches: list[list[str]]) -> str:
    """``main`` showing the first batch; each End key press appends the next."""
    return f"""<!DOCTYPE html>
<html lang="en">
  <head><meta charset="utf-8"><title>Activity</title></head>
  <body>
    <main><div id="feed"></div></main>
    <script>
      const batches = {json.dumps(batches)};
      const feed = document.getElementById('feed');
      let shown = 0;
      const next = () => {{
        if (shown >= batches.length) return;
        feed.insertAdjacentHTML('beforeend', batches[shown].join(''));
        shown++;
      }};
      next();
      document.addEventListener('keydown', event => {{
        if (event.key === 'End') setTimeout(next, 30);
      }});
    </script>
  </body>
</html>
"""


async def mounted(page: Page) -> list[dict]:
    return await PageSession(page).run_on_linkedin(activity_module._MOUNTED_POSTS)


class TestMountedPosts:
    async def test_each_post_is_read_with_its_author(self, dom_page):
        own = urn(1)
        reposted = urn(2)
        quoted, original = urn(3), urn(30)
        company = urn(4)
        unlinked = urn(5)
        html = feed(
            [
                [
                    post(
                        own,
                        "https://www.linkedin.com/in/ada-lovelace?miniProfileUrn=x",
                        "Ada Lovelace",
                        'Thanks <a href="/in/charles-babbage/">Charles Babbage</a>',
                    ),
                    repost(reposted, "/in/ada-lovelace/", "Ada Lovelace", "Engine"),
                    # A quote: the owner's post, with the original nested in it.
                    f'<div data-urn="{quoted}">'
                    f"{actor('/in/ada-lovelace/', 'Ada Lovelace')}<p>Worth a read</p>"
                    f"{post(original, '/in/charles-babbage/', 'Charles', 'Notes')}"
                    "</div>",
                    post(company, "/company/acme/", "Acme", "Launch"),
                    f'<div data-urn="{unlinked}"><p>No author link</p></div>',
                ]
            ]
        )
        await serve(dom_page, FEED_URL, html)

        posts = await mounted(dom_page)

        assert [(item["urn"], item["author"]) for item in posts] == [
            (own, "in:ada-lovelace"),
            (reposted, "in:charles-babbage"),
            (quoted, "in:ada-lovelace"),
            (company, "company:acme"),
            (unlinked, None),
        ]
        assert "Notes" in posts[2]["text"]

    async def test_an_encoded_slug_is_read_decoded_and_lowercased(self, dom_page):
        activity = urn(1)
        html = feed([[post(activity, "/in/J%C3%B6rg-M/", "Jörg M", "Hallo")]])
        await serve(dom_page, FEED_URL, html)

        assert (await mounted(dom_page))[0]["author"] == "in:jörg-m"

    async def test_posts_present_waits_for_a_post(self, dom_page):
        await serve(dom_page, FEED_URL, feed([[]]))
        session = PageSession(dom_page)

        assert await session.run_on_linkedin(activity_module._POSTS_PRESENT) is False
        await dom_page.evaluate(
            """activity => document.getElementById('feed').insertAdjacentHTML(
                'beforeend', `<div data-urn="${activity}"></div>`)""",
            urn(1),
        )
        assert await session.run_on_linkedin(activity_module._POSTS_PRESENT) is True


class TestTheFeedRead:
    async def test_keeps_the_owners_posts_by_date_past_a_pinned_one(self, dom_page):
        # The page's own pinned post is the only one mounted at first, and is
        # older than since_days; the owner's newer posts load on End.
        pinned, newest, middle, oldest = urn(200), urn(1), urn(2), urn(40)
        member_post = urn(1.5)
        html = feed(
            [
                [post(pinned, "/company/acme/", "Acme", "Pinned")],
                [
                    post(newest, "/company/acme/", "Acme", "Launch"),
                    repost(member_post, "/company/acme/", "Acme", "Shared"),
                    post(middle, "/company/acme/", "Acme", "Hiring"),
                ],
                [post(oldest, "/company/acme/", "Acme", "Old news")],
            ]
        )
        await serve(dom_page, COMPANY_FEED_URL, html)
        session = PageSession(dom_page)

        async def navigate(url: str) -> None:
            assert url == dom_page.url

        async def no_username() -> str:
            raise AssertionError("a company feed needs no username")

        reader = ActivityReader(
            session,
            cast(PageNavigator, SimpleNamespace(_navigate_to_page=navigate)),
            no_username,
            settle_timeout=1.0,
            poll_interval=0.05,
        )

        result = await reader.get_my_posts("acme", since_days=30)

        assert [item["urn"] for item in result["posts"]] == [newest, middle]
        assert "Launch" in result["posts"][0]["text"]
