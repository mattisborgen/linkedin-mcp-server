"""Read the posts of the user's own feed or a company page's, with their dates.

A post is a container whose ``data-urn`` starts with ``urn:li:activity:``, a
structural marker that holds in every UI language. Its publish time is decoded
from that URN rather than read from the rendered relative date ("2w"), which is
rounded and localized. The text, counts included, is returned as LinkedIn
renders it for the caller to read.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import logging
import re

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.destination import linkedin_element
from linkedin_mcp_server.linkedin.identifiers import (
    company_page_url,
    normalize_company_identifier,
    normalize_person_identifier,
    person_profile_url,
)
from linkedin_mcp_server.linkedin.link_metadata import Reference
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.session import NAV_DELAY, PageSession

logger = logging.getLogger(__name__)

PERSONAL_SOURCE = "personal"

# Without viewAsMember LinkedIn sends a page admin to the admin dashboard,
# which lists no posts.
COMPANY_POSTS_SUFFIX = "/posts/?feedView=all&viewAsMember=true"

ResolveOwnUsername = Callable[[], Awaitable[str]]

_ACTIVITY_URN = re.compile(r"urn:li:activity:([0-9]+)")

# Activity ids are Snowflake-style: the bits above the low 22 are the creation
# time in Unix milliseconds.
_TIMESTAMP_SHIFT = 22

_POSTS_PRESENT = """() => {
    const root = document.querySelector('main') || document.body;
    return root.querySelector('[data-urn^="urn:li:activity:"]') !== null;
}"""

# Every post mounted now, in page order. A post nested in another (the
# original inside a repost) is part of the outer one.
_MOUNTED_POSTS = """() => {
    const root = document.querySelector('main') || document.body;
    const selector = '[data-urn^="urn:li:activity:"]';
    const posts = [];
    for (const element of root.querySelectorAll(selector)) {
        if (element.parentElement && element.parentElement.closest(selector)) continue;
        posts.push({
            urn: element.getAttribute('data-urn'),
            text: (element.innerText || '').trim(),
        });
    }
    return posts;
}"""


def activity_published_at(urn: str) -> datetime | None:
    """When an ``urn:li:activity:<id>`` was created, in UTC; None without an id."""
    match = _ACTIVITY_URN.fullmatch(urn)
    if match is None:
        return None
    milliseconds = int(match.group(1)) >> _TIMESTAMP_SHIFT
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def normalize_posts_source(source: str) -> str | None:
    """None for the user's own posts, otherwise the company's /company/ slug.

    Raises:
        InvalidReferenceError: When *source* is neither ``"personal"`` nor a
            company slug or URL.
    """
    if source.strip().lower() == PERSONAL_SOURCE:
        return None
    return normalize_company_identifier(source)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class _Post:
    urn: str
    published: datetime
    text: str


class _Harvest:
    """Every post the feed has shown, by URN, in the order it first appeared.

    Kept across rounds because the feed unmounts posts as it grows, so a single
    read at the end would miss the earliest ones.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._posts: dict[str, _Post] = {}

    @property
    def seen(self) -> int:
        """Distinct posts seen, dated or not: the signal that a batch loaded."""
        return len(self._seen)

    def add(self, mounted: Any) -> None:
        for item in mounted or ():
            urn = item.get("urn") if isinstance(item, dict) else None
            if not isinstance(urn, str) or not urn:
                continue
            text = item.get("text") or ""
            self._seen.add(urn)
            known = self._posts.get(urn)
            if known is not None:
                # An unmounting post can leave an emptied shell behind.
                if len(text) > len(known.text):
                    known.text = text
                continue
            published = activity_published_at(urn)
            if published is None:
                logger.debug("Skipping a post whose URN carries no time: %s", urn)
                continue
            self._posts[urn] = _Post(urn, published, text)

    def since(self, cutoff: datetime) -> list[_Post]:
        return [post for post in self._posts.values() if post.published >= cutoff]

    def passed(self, cutoff: datetime) -> bool:
        """Whether the last post loaded is older than *cutoff*.

        The feed is newest first, so everything below it is older still. The
        last post rather than the oldest, because a pinned post sits at the
        top whatever its age.
        """
        if not self._posts:
            return False
        return next(reversed(self._posts.values())).published < cutoff


class ActivityReader:
    """Read the posts of the user's own feed or a company page's feed."""

    def __init__(
        self,
        session: PageSession,
        navigator: PageNavigator,
        resolve_own_username: ResolveOwnUsername,
        *,
        settle_timeout: float = 3.0,
        poll_interval: float = 0.25,
        max_rounds: int = 60,
    ):
        self._session = session
        self._navigator = navigator
        self._resolve_own_username = resolve_own_username
        self._settle_timeout = settle_timeout
        self._poll_interval = poll_interval
        self._max_rounds = max_rounds

    async def get_my_posts(
        self, source: str = PERSONAL_SOURCE, since_days: int = 30, limit: int = 50
    ) -> dict[str, Any]:
        """List the posts of the last *since_days* days, newest first."""
        url = await self._feed_url(source)
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()
        try:
            await self._session.page.wait_for_function(_POSTS_PRESENT, timeout=10000)
        except PlaywrightTimeoutError:
            logger.debug("No post appeared on %s", url)
        await self._session.dismiss_modal()

        cutoff = _utc_now() - timedelta(days=since_days)
        harvest = _Harvest()
        await self._load(harvest, cutoff, limit)
        logger.debug("%s: %d posts seen (limit %d)", url, harvest.seen, limit)
        posts = sorted(
            harvest.since(cutoff),
            key=lambda post: (post.published, post.urn),
            reverse=True,
        )[:limit]
        return _result(url, posts, saw_posts=harvest.seen > 0)

    async def _feed_url(self, source: str) -> str:
        company = normalize_posts_source(source)
        if company is not None:
            return company_page_url(company, COMPANY_POSTS_SUFFIX)
        username = normalize_person_identifier(
            await self._resolve_own_username(), allow_self_alias=True
        )
        await self._session.delay(NAV_DELAY)
        return person_profile_url(username, "/recent-activity/all/")

    async def _load(self, harvest: _Harvest, cutoff: datetime, limit: int) -> None:
        """Load posts until *limit* are in range, the feed passes *cutoff*, or it ends."""
        harvest.add(await self._mounted())
        for _ in range(self._max_rounds):
            if (
                not harvest.seen
                or len(harvest.since(cutoff)) >= limit
                or harvest.passed(cutoff)
            ):
                return
            if not await self._load_more(harvest):
                # One retry: the batch may still have been on its way.
                if not await self._load_more(harvest):
                    return

    async def _load_more(self, harvest: _Harvest) -> bool:
        """Press End once and wait for a post the harvest has not seen."""
        previous = harvest.seen
        # Each press fetches the next batch, so it is paced like a short page load.
        await self._session.pace(0.4)
        await self._press_end()
        deadline = self._session.monotonic() + self._settle_timeout
        while self._session.monotonic() < deadline:
            await self._session.delay(self._poll_interval)
            harvest.add(await self._mounted())
            if harvest.seen > previous:
                return True
        return False

    async def _press_end(self) -> None:
        # A real key press, because the feed loads its next batch on End and
        # ignores script scrolling and synthetic events. Pressed through a body
        # handle rather than page.keyboard, so it lands only in a LinkedIn
        # document.
        async with linkedin_element(self._session.page.locator("body")) as body:
            await body.press("End")

    async def _mounted(self) -> Any:
        return await self._session.run_on_linkedin(_MOUNTED_POSTS)


def _result(url: str, posts: list[_Post], *, saw_posts: bool) -> dict[str, Any]:
    entries = [
        {
            "urn": post.urn,
            "url": f"/feed/update/{post.urn}/",
            "published_at": _iso(post.published),
            "text": post.text,
        }
        for post in posts
    ]
    result: dict[str, Any] = {"url": url, "sections": {}}
    if entries:
        result["sections"]["posts"] = "\n\n".join(
            f"--- {entry['urn']} | published {entry['published_at']} ---\n"
            f"{entry['text']}"
            for entry in entries
        )
        result["references"] = {"posts": [_reference(entry) for entry in entries]}
    elif not saw_posts:
        result["section_errors"] = {"posts": _no_posts_section_error()}
    result["posts"] = entries
    return result


def _reference(entry: dict[str, str]) -> Reference:
    reference: Reference = {"kind": "feed_post", "url": entry["url"]}
    first_line = next(
        (line.strip() for line in entry["text"].splitlines() if line.strip()), ""
    )
    if first_line:
        reference["text"] = first_line
    return reference


def _no_posts_section_error() -> dict[str, str]:
    """Explain an empty list, which also describes a page that did not render."""
    return {
        "error_type": "no_posts_found",
        "error_message": (
            "The page showed no posts. Either the feed has none, or LinkedIn "
            "changed how it marks posts and they were not recognized."
        ),
    }
