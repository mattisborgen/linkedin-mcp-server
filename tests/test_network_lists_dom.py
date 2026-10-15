"""Browser-DOM tests for the connection and invitation list programs.

The unit suite mocks ``page.evaluate``, so the three programs in
``linkedin/network.py`` never execute there: the profile count, the scroll of
every scrollable region in ``<main>`` and the per-card reference read are
asserted as call arguments and nothing else. These cases run them in headless
chromium, and run the whole list read once against a list that only grows when
its own container is scrolled.

Every fixture drives a synthetic container, so each case is a claim about the
algorithm rather than about LinkedIn's markup. Skipped automatically when
chromium is not installed; run locally after
``uv run patchright install chromium --no-shell``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import json

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.linkedin import network as network_module
from linkedin_mcp_server.linkedin.content import PageContentReader
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.network import (
    CONNECTIONS_URL,
    SENT_INVITATIONS_URL,
    NetworkReader,
)
from linkedin_mcp_server.linkedin.session import PageSession

#: CI uses ``--dist loadgroup``. Keep every test that launches Chromium on one
#: worker so browser startups cannot compete with the DOM cases' wall-clock
#: timers.
#: Without that distribution mode the group mark is inert.
pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]


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


def connection(slug: str, name: str) -> dict[str, str]:
    """A connection card: photo link, name link, headline and date."""
    return {
        "slug": slug,
        "html": (
            f'<a href="https://www.linkedin.com/in/{slug}/?trk=x"><img alt=""></a>'
            f'<a href="/in/{slug}/"><span>{name}</span></a>'
            f"<p>Analyst</p><p>Connected on October 1, 2026</p>"
        ),
    }


def invitation(slug: str, name: str) -> dict[str, str]:
    """An invitation card: only the photo links to the profile."""
    return {
        "slug": slug,
        "html": (
            f"<div><p>{name}</p><p>Engineer</p></div>"
            f'<a href="/in/{slug}/"><img alt=""></a>'
            f"<p>Sent 2 weeks ago</p><button>Withdraw</button>"
        ),
    }


def profile_list(
    cards: list[dict[str, str]], *, initial: int, batch: int, outside: str = ""
) -> str:
    """``main`` holding a list that scrolls in its own container.

    *initial* cards are mounted at first, and every scroll of the container
    appends the next *batch* 30ms later. The window itself never scrolls, so a
    read that scrolls only the window loads nothing. *outside* is markup after
    ``main``.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
  <head><meta charset="utf-8"><title>Network</title></head>
  <body>
    <main style="height:300px;overflow:hidden">
      <ul id="list" style="height:200px;overflow-y:auto;margin:0"></ul>
    </main>
    {outside}
    <script>
      const cards = {json.dumps(cards)};
      const list = document.getElementById('list');
      let shown = 0;
      const add = count => {{
        for (const card of cards.slice(shown, shown + count)) {{
          const li = document.createElement('li');
          li.style.height = '80px';
          li.innerHTML = card.html;
          list.appendChild(li);
        }}
        shown = Math.min(cards.length, shown + count);
      }};
      add({initial});
      let pending = false;
      list.addEventListener('scroll', () => {{
        if (pending || shown >= cards.length) return;
        pending = true;
        setTimeout(() => {{ add({batch}); pending = false; }}, 30);
      }});
    </script>
  </body>
</html>
"""


def _reader(page: Page) -> NetworkReader:
    session = PageSession(page)

    async def navigate(url: str) -> None:
        # Already served; a real navigation would only reload the fixture.
        assert url == page.url

    return NetworkReader(
        session,
        cast(PageNavigator, SimpleNamespace(_navigate_to_page=navigate)),
        PageContentReader(session),
        settle_timeout=0.5,
        poll_interval=0.05,
    )


async def run(page: Page, program: str) -> Any:
    return await PageSession(page).run_on_linkedin(program)


class TestProfileCount:
    async def test_counts_each_profile_in_main_once(self, dom_page):
        cards = [connection("ada-lovelace", "Ada"), connection("charles", "Charles")]
        outside = '<a href="/in/someone-else/">Outside main</a>'
        await serve(
            dom_page,
            CONNECTIONS_URL,
            profile_list(cards, initial=2, batch=0, outside=outside),
        )

        # Two links per card, one query string, one link outside <main>.
        assert await run(dom_page, network_module._COUNT_PROFILES) == 2


class TestProfileReferences:
    async def test_a_card_is_named_by_its_link_or_its_first_line(self, dom_page):
        cards = [
            connection("ada-lovelace", "Ada Lovelace"),
            invitation("charles-babbage", "Charles Babbage"),
            invitation("mary-somerville", "Mary Somerville"),
        ]
        await serve(
            dom_page, SENT_INVITATIONS_URL, profile_list(cards, initial=3, batch=0)
        )

        profiles = await run(dom_page, network_module._PROFILE_LINKS)

        # The invitation cards link only their photo, so each name comes from
        # its own card, never from the card beside it.
        assert profiles == [
            {"slug": "ada-lovelace", "text": "Ada Lovelace"},
            {"slug": "charles-babbage", "text": "Charles Babbage"},
            {"slug": "mary-somerville", "text": "Mary Somerville"},
        ]


class TestTheListRead:
    async def test_scrolls_the_lists_own_container_until_the_limit(self, dom_page):
        cards = [
            connection(f"person-{index}", f"Person {index}") for index in range(12)
        ]
        await serve(dom_page, CONNECTIONS_URL, profile_list(cards, initial=4, batch=4))

        result = await _reader(dom_page).get_connections(limit=10)

        # One batch lands per scroll, so stopping before the limit, or not
        # reaching the container at all, leaves fewer than ten.
        urls = [reference["url"] for reference in result["references"]["connections"]]
        assert urls == [f"/in/person-{index}/" for index in range(10)]
        assert "Person 9" in result["sections"]["connections"]

    async def test_stops_when_the_list_ends(self, dom_page):
        cards = [invitation(f"person-{index}", f"Person {index}") for index in range(6)]
        await serve(
            dom_page, SENT_INVITATIONS_URL, profile_list(cards, initial=4, batch=4)
        )

        result = await _reader(dom_page).get_sent_invitations(limit=100)

        references = result["references"]["sent_invitations"]
        assert [reference["text"] for reference in references] == [
            f"Person {index}" for index in range(6)
        ]
