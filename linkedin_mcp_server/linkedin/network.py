"""Read the authenticated user's own network lists.

Both lists, connections and sent invitations, scroll inside a container in
``<main>`` rather than the window, so ``scroll_to_bottom`` leaves them at their
first batch. Loading is measured by distinct ``/in/`` profile anchors, a
structural signal that holds in every UI language; the dates ("Connected on",
"Sent 2 weeks ago") stay in the returned text for the caller to read.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.linkedin.content import PageContentReader
from linkedin_mcp_server.linkedin.contracts import RATE_LIMITED_SECTION_TEXT
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.session import PageSession
from linkedin_mcp_server.linkedin.text import strip_linkedin_noise

logger = logging.getLogger(__name__)

CONNECTIONS_URL = "https://www.linkedin.com/mynetwork/invite-connect/connections/"
SENT_INVITATIONS_URL = "https://www.linkedin.com/mynetwork/invitation-manager/sent/"

# Distinct profile links inside <main>; the page's own profile card, if any,
# is outside <main>.
_COUNT_PROFILES = """() => {
    const main = document.querySelector('main');
    if (!main) return 0;
    const slugs = new Set();
    for (const a of main.querySelectorAll('a[href*="/in/"]')) {
        const match = a.getAttribute('href').match(/\\/in\\/([^/?#]+)/);
        if (match) slugs.add(match[1]);
    }
    return slugs.size;
}"""

# Scroll every scrollable region in <main> (and <main> itself), plus the
# window: which one owns the list differs between the two pages and changes
# with LinkedIn's layout, and scrolling one that holds nothing loads nothing.
_SCROLL_LISTS = """() => {
    const main = document.querySelector('main');
    if (!main) return false;
    const scrollable = element => {
        const style = window.getComputedStyle(element);
        return (style.overflowY === 'auto' || style.overflowY === 'scroll')
            && element.scrollHeight > element.clientHeight + 20;
    };
    for (const element of [main, ...main.querySelectorAll('*')].filter(scrollable)) {
        element.scrollTop = element.scrollHeight;
    }
    window.scrollTo(0, document.body.scrollHeight);
    return true;
}"""


# Each distinct profile in <main>, in page order, labelled by its card. On the
# invitations page the profile link wraps only the photo, so the anchor itself
# has no text. The card is the largest ancestor that still links to that one
# profile only, a structural boundary; its first text line is the name.
_PROFILE_LINKS = """() => {
    const main = document.querySelector('main');
    if (!main) return [];
    const slugOf = a => {
        const match = (a.getAttribute('href') || '').match(/\\/in\\/([^/?#]+)/);
        return match ? match[1] : null;
    };
    const linksOnlyTo = (element, slug) => {
        const slugs = new Set();
        for (const a of element.querySelectorAll('a[href*="/in/"]')) slugs.add(slugOf(a));
        return slugs.size === 1 && slugs.has(slug);
    };
    const firstLine = element =>
        (element.innerText || '').split('\\n').map(line => line.trim()).find(Boolean) || '';
    const seen = new Set();
    const profiles = [];
    for (const a of main.querySelectorAll('a[href*="/in/"]')) {
        const slug = slugOf(a);
        if (!slug || seen.has(slug)) continue;
        seen.add(slug);
        let card = a;
        while (card.parentElement && card.parentElement !== main
               && linksOnlyTo(card.parentElement, slug)) {
            card = card.parentElement;
        }
        profiles.push({ slug, text: firstLine(a) || firstLine(card) });
    }
    return profiles;
}"""


class NetworkReader:
    """Read the user's connections and pending sent invitations."""

    def __init__(
        self,
        session: PageSession,
        navigator: PageNavigator,
        content: PageContentReader,
        *,
        settle_timeout: float = 3.0,
        poll_interval: float = 0.25,
        max_rounds: int = 60,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content
        self._settle_timeout = settle_timeout
        self._poll_interval = poll_interval
        self._max_rounds = max_rounds

    async def get_connections(self, limit: int = 50) -> dict[str, Any]:
        """List the most recently added connections, newest first."""
        return await self._read_profile_list(CONNECTIONS_URL, "connections", limit)

    async def get_sent_invitations(self, limit: int = 100) -> dict[str, Any]:
        """List pending sent invitations, newest first."""
        return await self._read_profile_list(
            SENT_INVITATIONS_URL, "sent_invitations", limit
        )

    async def _read_profile_list(
        self, url: str, section_name: str, limit: int
    ) -> dict[str, Any]:
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()
        try:
            await self._session.page.wait_for_function(
                f"() => ({_COUNT_PROFILES})() > 0", timeout=10000
            )
        except PlaywrightTimeoutError:
            # An empty list is a real answer: no connections or no pending invites.
            logger.debug("No profile links appeared on %s", url)
        await self._session.dismiss_modal()

        loaded = await self._load_profiles(limit)
        logger.debug("%s: %d profiles loaded (limit %d)", section_name, loaded, limit)

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]
        result: dict[str, Any] = {"url": url, "sections": {}}
        if not raw:
            return result
        cleaned = strip_linkedin_noise(raw)
        if not cleaned and raw.strip():
            result["sections"][section_name] = RATE_LIMITED_SECTION_TEXT
            return result
        result["sections"][section_name] = cleaned
        references = await self._profile_references(limit)
        if references:
            result["references"] = {section_name: references}
        return result

    async def _load_profiles(self, limit: int) -> int:
        """Scroll until *limit* profiles are loaded or the list stops growing."""
        count = await self._count()
        for _ in range(self._max_rounds):
            if count >= limit:
                break
            # Each scroll fetches the next batch, so it is paced like a short page load.
            await self._session.pace(0.4)
            await self._session.run_on_linkedin(_SCROLL_LISTS)
            grown = await self._wait_for_growth(count)
            if grown == count:
                # One retry: the list may have moved while the first wait ran.
                await self._session.run_on_linkedin(_SCROLL_LISTS)
                grown = await self._wait_for_growth(count)
                if grown == count:
                    break
            count = grown
        return count

    async def _wait_for_growth(self, previous: int) -> int:
        """Poll until the profile count passes *previous* or the settle time ends."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settle_timeout
        count = previous
        while loop.time() < deadline:
            await self._session.delay(self._poll_interval)
            count = await self._count()
            if count > previous:
                return count
        return count

    async def _count(self) -> int:
        return int(await self._session.run_on_linkedin(_COUNT_PROFILES) or 0)

    async def _profile_references(self, limit: int) -> list[dict[str, str]]:
        profiles = await self._session.run_on_linkedin(_PROFILE_LINKS) or []
        references = []
        for profile in profiles[:limit]:
            reference = {"kind": "person", "url": f"/in/{profile['slug']}/"}
            if profile.get("text"):
                reference["text"] = profile["text"]
            references.append(reference)
        return references
