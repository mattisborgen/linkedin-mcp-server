"""Read the posts of the user's own feed or a company page's, with their dates.

A post is a container whose ``data-urn`` starts with ``urn:li:activity:``, a
structural marker that holds in every UI language. Its publish time is decoded
from that URN rather than read from the rendered relative date ("2w"), which is
rounded and localized. The text, counts included, is returned as LinkedIn
renders it for the caller to read.

Only the feed owner's own posts are returned. A repost carries the original
author's avatar link, and a card whose avatar links to someone else is not the
owner's post, whatever the line above it says in the page's language.
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

# Every post mounted now, in page order, with its author. A post nested in
# another (the original inside a repost) is part of the outer one.
#
# The author is the first /in/ or /company/ link in the card that holds an
# image or no text: the avatar beside the name of whoever wrote the post. DOM
# rather than text, because only the link target names the author in every UI
# language; the "reposted this" line above a repost is a sentence. Name links,
# @-mentions and the reposter's name in that line all carry text and are
# passed over. "in:<slug>" or "company:<slug>", or null when no avatar link.
_MOUNTED_POSTS = """() => {
    const root = document.querySelector('main') || document.body;
    const selector = '[data-urn^="urn:li:activity:"]';
    const identity = /\\/(in|company)\\/([^/?#]+)/;
    const authorOf = card => {
        for (const a of card.querySelectorAll('a[href*="/in/"], a[href*="/company/"]')) {
            const match = (a.getAttribute('href') || '').match(identity);
            if (!match) continue;
            const avatar = a.querySelector('img') !== null
                || !(a.textContent || '').trim();
            if (!avatar) continue;
            let slug = match[2];
            try { slug = decodeURIComponent(slug); } catch (error) {}
            return `${match[1]}:${slug.toLowerCase()}`;
        }
        return null;
    };
    const posts = [];
    for (const element of root.querySelectorAll(selector)) {
        if (element.parentElement && element.parentElement.closest(selector)) continue;
        posts.push({
            urn: element.getAttribute('data-urn'),
            text: (element.innerText || '').trim(),
            author: authorOf(element),
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


@dataclass(frozen=True, slots=True)
class FeedOwner:
    """Whose feed is read: ``kind`` is "in" or "company", ``slug`` its id.

    ``slug`` is None when the user's own username could not be resolved.
    """

    kind: str
    slug: str | None

    def wrote(self, author: str | None) -> bool:
        """Whether a card with this *author* is the owner's own post.

        No author is no evidence either way, and the card is kept. An author
        of the other kind is someone else. One of the same kind is compared
        only when both sides are slugs: a member id (``ACoA...``) or a numeric
        company id names the same account as a slug, and cannot be told apart
        from it here.
        """
        if not author:
            return True
        kind, _, slug = author.partition(":")
        if kind != self.kind:
            return False
        if self.slug is None or _opaque(kind, slug) or _opaque(kind, self.slug):
            return True
        return slug.casefold() == self.slug.casefold()


def _opaque(kind: str, slug: str) -> bool:
    if kind == "company":
        return slug.isdigit()
    return slug.casefold().startswith("acoa")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class _Post:
    urn: str
    published: datetime
    text: str
    author: str | None


class _Harvest:
    """Every dated post the feed has shown, by URN, in the order it first appeared.

    Kept across rounds because the feed unmounts posts as it grows, so a single
    read at the end would miss the earliest ones.
    """

    def __init__(self, owner: FeedOwner) -> None:
        self._owner = owner
        self._seen: set[str] = set()
        self._posts: dict[str, _Post] = {}

    @property
    def seen(self) -> int:
        """Distinct posts seen, anyone's, dated or not: the signal that a batch loaded."""
        return len(self._seen)

    def add(self, mounted: Any) -> None:
        for item in mounted or ():
            urn = item.get("urn") if isinstance(item, dict) else None
            if not isinstance(urn, str) or not urn:
                continue
            text = item.get("text") or ""
            author = item.get("author") or None
            self._seen.add(urn)
            known = self._posts.get(urn)
            if known is not None:
                # An unmounting post can leave an emptied shell behind.
                if len(text) > len(known.text):
                    known.text = text
                if known.author is None:
                    known.author = author
                continue
            published = activity_published_at(urn)
            if published is None:
                logger.debug("Skipping a post whose URN carries no time: %s", urn)
                continue
            self._posts[urn] = _Post(urn, published, text, author)

    def own(self) -> list[_Post]:
        """The owner's posts, in feed order."""
        return [post for post in self._posts.values() if self._owner.wrote(post.author)]

    @property
    def others(self) -> int:
        return len(self._posts) - len(self.own())

    def settled(self) -> list[_Post]:
        """The owner's posts in feed order, from the first one known to be in order.

        The feed is newest first except for pinned posts, which sit at the
        top whatever their age. A post older than the one after it is such a
        post. The last post of a run that is still rising has no successor
        to be compared with yet, so until one loads nothing is settled: a
        pinned post that is the only one mounted, or the newest one loaded so
        far, must not end the read or count toward the limit.
        """
        own = self.own()
        first = 0
        while first + 1 < len(own) and own[first].published < own[first + 1].published:
            first += 1
        if first + 1 >= len(own):
            return []
        return own[first:]

    def since(self, cutoff: datetime) -> list[_Post]:
        """The owner's posts published at or after *cutoff*, pinned ones too."""
        return [post for post in self.own() if post.published >= cutoff]

    def enough(self, cutoff: datetime, limit: int) -> bool:
        """Whether *limit* settled posts are in range."""
        settled = self.settled()
        return sum(post.published >= cutoff for post in settled) >= limit

    def passed(self, cutoff: datetime) -> bool:
        """Whether the last settled post loaded is older than *cutoff*.

        The feed is newest first below its pinned posts, so everything after
        it is older still.
        """
        settled = self.settled()
        return bool(settled) and settled[-1].published < cutoff


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
        """List the owner's posts of the last *since_days* days, newest first."""
        url, owner = await self._feed(source)
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()
        try:
            await self._session.page.wait_for_function(_POSTS_PRESENT, timeout=10000)
        except PlaywrightTimeoutError:
            logger.debug("No post appeared on %s", url)
        await self._session.dismiss_modal()

        cutoff = _utc_now() - timedelta(days=since_days)
        harvest = _Harvest(owner)
        await self._load(harvest, cutoff, limit)
        logger.debug(
            "%s: %d posts seen, %d by someone else (limit %d)",
            url,
            harvest.seen,
            harvest.others,
            limit,
        )
        # By publish time rather than page order, so a pinned post takes its
        # own place instead of the top.
        posts = sorted(
            harvest.since(cutoff),
            key=lambda post: (post.published, post.urn),
            reverse=True,
        )[:limit]
        return _result(url, posts, saw_posts=harvest.seen > 0)

    async def _feed(self, source: str) -> tuple[str, FeedOwner]:
        """The feed's address and whose posts it holds."""
        company = normalize_posts_source(source)
        if company is not None:
            return company_page_url(company, COMPANY_POSTS_SUFFIX), FeedOwner(
                "company", company
            )
        username = normalize_person_identifier(
            await self._resolve_own_username(), allow_self_alias=True
        )
        await self._session.delay(NAV_DELAY)
        owner = FeedOwner("in", None if username == "me" else username)
        return person_profile_url(username, "/recent-activity/all/"), owner

    async def _load(self, harvest: _Harvest, cutoff: datetime, limit: int) -> None:
        """Load posts until *limit* are in range, the feed passes *cutoff*, or it ends."""
        harvest.add(await self._mounted())
        for _ in range(self._max_rounds):
            if (
                not harvest.seen
                or harvest.enough(cutoff, limit)
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
        # Each press, the retry's too, fetches the next batch: paced like a page load.
        await self._session.pace()
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
