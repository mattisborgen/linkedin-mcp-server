"""Tests for reading the user's own or a company page's posts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from patchright.async_api import Page

from linkedin_mcp_server.linkedin import activity as activity_module
from linkedin_mcp_server.linkedin import session as session_module
from linkedin_mcp_server.linkedin.activity import (
    ActivityReader,
    FeedOwner,
    activity_published_at,
)
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.session import PageSession

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
OWNER = "in:ada-lovelace"
PAUSE = 3.0


def _post(
    days_ago: float, text: str | None = None, *, author: str | None = OWNER
) -> dict[str, Any]:
    """A mounted post published *days_ago* before NOW, by the feed's owner."""
    milliseconds = int((NOW - timedelta(days=days_ago)).timestamp() * 1000)
    urn = f"urn:li:activity:{milliseconds << 22 | 99}"
    return {
        "urn": urn,
        "text": text if text is not None else f"Post {days_ago}",
        "author": author,
    }


class _Feed:
    """A post feed that loads one batch per End press, as LinkedIn's does.

    *mounted* keeps only that many of the loaded posts in the page, the newest,
    the way a virtualized list unmounts what scrolled away; with *emptied* they
    stay but lose their text instead. *ignored* names presses (1-based) that
    load nothing.
    """

    def __init__(
        self,
        batches: list[list[dict[str, str]]],
        *,
        mounted: int | None = None,
        emptied: bool = False,
        ignored: Iterable[int] = (),
    ):
        self.batches = batches
        self.loaded = 1
        self.presses = 0
        self.mounted = mounted
        self.emptied = emptied
        self.ignored = set(ignored)
        self.url = "https://www.linkedin.com/in/ada-lovelace/recent-activity/all/"
        self.log: list[Any] = []

    async def wait_for_function(self, expression: str, *, timeout: int) -> None:
        return None

    async def evaluate(self, expression: str) -> list[dict[str, str]]:
        assert "closest(selector)" in expression
        posts = [dict(post) for batch in self.batches[: self.loaded] for post in batch]
        if self.mounted is None:
            return posts
        if self.emptied:
            for post in posts[: -self.mounted]:
                post["text"] = ""
            return posts
        return posts[-self.mounted :]

    def locator(self, selector: str) -> SimpleNamespace:
        assert selector == "body"
        feed = self

        class Handle:
            async def evaluate(self, expression: str) -> str:
                return feed.url

            async def press(self, key: str) -> None:
                assert key == "End"
                feed.presses += 1
                feed.log.append("press")
                if feed.presses not in feed.ignored and feed.loaded < len(feed.batches):
                    feed.loaded += 1

            async def dispose(self) -> None:
                return None

        async def element_handle(**_kwargs: Any) -> Handle:
            return Handle()

        return SimpleNamespace(element_handle=element_handle)


@pytest.fixture(autouse=True)
def page_boundaries(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A clock that moves only when the reader waits, and a fixed today."""
    clock = SimpleNamespace(now=0.0, sleeps=[])

    async def sleep(seconds: float) -> None:
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(session_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(session_module, "detect_rate_limit", AsyncMock())
    monkeypatch.setattr(
        session_module, "handle_modal_close", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(activity_module, "_utc_now", lambda: NOW)
    return clock


def _reader(
    feed: _Feed, *, pause: tuple[float, float] = (0.0, 0.0)
) -> tuple[ActivityReader, AsyncMock, AsyncMock]:
    session = PageSession(cast(Page, feed), pause)
    navigate = AsyncMock()
    navigator = cast(PageNavigator, SimpleNamespace(_navigate_to_page=navigate))
    resolve = AsyncMock(return_value="ada-lovelace")
    return ActivityReader(session, navigator, resolve), navigate, resolve


def _urns(result: dict[str, Any]) -> list[str]:
    return [post["urn"] for post in result["posts"]]


class TestPublishTime:
    def test_is_decoded_from_the_urn(self):
        published = activity_published_at("urn:li:activity:7152605457802253959")

        assert published == datetime(2024, 1, 15, 10, 20, 30, 123000, tzinfo=UTC)

    @pytest.mark.parametrize(
        "urn",
        ["urn:li:activity:", "urn:li:activity:12a", "urn:li:share:7152605457802253959"],
    )
    def test_a_urn_without_an_activity_id_has_none(self, urn: str):
        assert activity_published_at(urn) is None


class TestWhenTheReadStops:
    async def test_at_the_limit(self):
        batches = [[_post(1), _post(2)], [_post(3), _post(4)], [_post(5), _post(6)]]
        feed = _Feed(batches)
        reader, _navigate, _resolve = _reader(feed)

        result = await reader.get_my_posts(limit=4)

        assert feed.presses == 1
        assert _urns(result) == [post["urn"] for post in batches[0] + batches[1]]

    async def test_once_the_last_post_loaded_is_older_than_since_days(self):
        batches = [[_post(1), _post(2)], [_post(3), _post(40)], [_post(41)]]
        feed = _Feed(batches)
        reader, _navigate, _resolve = _reader(feed)

        result = await reader.get_my_posts(since_days=30)

        assert feed.presses == 1
        assert _urns(result) == [post["urn"] for post in batches[0] + batches[1][:1]]

    async def test_not_at_a_pinned_post_older_than_since_days(self):
        # A pinned post sits at the top whatever its age.
        pinned, newer = _post(200), [_post(1), _post(2)]
        feed = _Feed([[pinned, newer[0]], [newer[1]]])
        reader, _navigate, _resolve = _reader(feed)

        result = await reader.get_my_posts(since_days=30)

        assert _urns(result) == [post["urn"] for post in newer]

    async def test_after_one_retry_when_nothing_more_loads(self):
        batch = [_post(1), _post(2)]
        feed = _Feed([batch])
        reader, _navigate, _resolve = _reader(feed)

        result = await reader.get_my_posts()

        assert feed.presses == 2
        assert _urns(result) == [post["urn"] for post in batch]

    async def test_not_before_a_batch_that_answers_the_retry(self):
        first, second = _post(1), _post(2)
        feed = _Feed([[first], [second]], ignored={1})
        reader, _navigate, _resolve = _reader(feed)

        result = await reader.get_my_posts()

        assert _urns(result) == [first["urn"], second["urn"]]

    async def test_before_pressing_on_a_page_with_no_posts(self):
        feed = _Feed([[]])
        reader, _navigate, _resolve = _reader(feed)

        result = await reader.get_my_posts()

        assert feed.presses == 0
        assert result["posts"] == []
        assert result["section_errors"]["posts"]["error_type"] == "no_posts_found"


class TestPacing:
    async def test_every_press_waits_the_full_pause_the_retry_too(
        self, monkeypatch: pytest.MonkeyPatch, page_boundaries: SimpleNamespace
    ):
        # One batch, then nothing: a press that loads, one that does not, and
        # the retry after it.
        feed = _Feed([[_post(1)], [_post(2)]])

        async def sleep(seconds: float) -> None:
            feed.log.append(("sleep", seconds))
            page_boundaries.now += seconds

        monkeypatch.setattr(session_module.asyncio, "sleep", sleep)
        reader, _navigate, _resolve = _reader(feed, pause=(PAUSE, PAUSE))

        await reader.get_my_posts()

        assert feed.presses == 3
        for index, event in enumerate(feed.log):
            if event == "press":
                assert feed.log[index - 1] == ("sleep", PAUSE), feed.log


class TestWhosePostsAreKept:
    async def test_a_repost_of_someone_elses_post_is_left_out(self):
        # Before the owner's older post in the feed, and still no place taken:
        # the limit counts the owner's posts only.
        newest, older = _post(1), _post(3)
        repost = _post(2, "Charles Babbage\nThe engine", author="in:charles-babbage")
        reader, _navigate, _resolve = _reader(_Feed([[newest, repost, older]]))

        result = await reader.get_my_posts(limit=2)

        assert _urns(result) == [newest["urn"], older["urn"]]

    async def test_a_post_with_no_author_link_read_is_kept(self):
        post = _post(1, author=None)
        reader, _navigate, _resolve = _reader(_Feed([[post]]))

        result = await reader.get_my_posts()

        assert _urns(result) == [post["urn"]]

    async def test_on_a_company_page_a_members_post_is_left_out(self):
        launch = _post(1, author="company:acme")
        shared = _post(2, author="in:ada-lovelace")
        partner = _post(3, author="company:analytical-engine")
        reader, _navigate, _resolve = _reader(_Feed([[launch, shared, partner]]))

        result = await reader.get_my_posts("acme")

        assert _urns(result) == [launch["urn"]]

    @pytest.mark.parametrize(
        ("owner", "author", "wrote"),
        [
            (FeedOwner("in", "ada-lovelace"), "in:ada-lovelace", True),
            (FeedOwner("in", "Ada-Lovelace"), "in:ada-lovelace", True),
            (FeedOwner("in", "ada-lovelace"), "in:charles-babbage", False),
            (FeedOwner("in", "ada-lovelace"), "company:acme", False),
            # A member id names an account a slug may also name.
            (FeedOwner("in", "ada-lovelace"), "in:acoaab1234", True),
            (FeedOwner("in", None), "in:charles-babbage", True),
            (FeedOwner("in", None), "company:acme", False),
            (FeedOwner("company", "acme"), "company:acme", True),
            (FeedOwner("company", "acme"), "company:12345", True),
            (FeedOwner("company", "acme"), "company:globex", False),
            (FeedOwner("company", "acme"), "in:ada-lovelace", False),
            (FeedOwner("company", "acme"), None, True),
        ],
    )
    def test_who_counts_as_the_owner(self, owner: FeedOwner, author, wrote: bool):
        assert owner.wrote(author) is wrote


class TestPinnedPosts:
    """A pinned post sits at the top of the feed whatever its age."""

    async def test_one_mounted_alone_and_out_of_range_does_not_end_the_read(self):
        pinned, newer = _post(200), [_post(1), _post(2)]
        reader, _navigate, _resolve = _reader(_Feed([[pinned], newer]))

        result = await reader.get_my_posts(since_days=30)

        assert _urns(result) == [post["urn"] for post in newer]

    async def test_one_in_range_does_not_take_the_place_of_a_newer_post(self):
        pinned, newest, next_newest = _post(10), _post(1), _post(2)
        reader, _navigate, _resolve = _reader(_Feed([[pinned], [newest, next_newest]]))

        result = await reader.get_my_posts(limit=1)

        assert _urns(result) == [newest["urn"]]

    async def test_one_in_range_is_placed_by_its_date(self):
        pinned, first, second, third = _post(1.5), _post(1), _post(2), _post(3)
        reader, _navigate, _resolve = _reader(_Feed([[pinned, first, second], [third]]))

        result = await reader.get_my_posts(limit=3)

        assert _urns(result) == [first["urn"], pinned["urn"], second["urn"]]


class TestWhatIsKept:
    async def test_posts_unmounted_before_the_end(self):
        batches = [[_post(1), _post(2)], [_post(3), _post(4)], [_post(5), _post(6)]]
        reader, _navigate, _resolve = _reader(_Feed(batches, mounted=2))

        result = await reader.get_my_posts()

        assert _urns(result) == [post["urn"] for batch in batches for post in batch]

    async def test_the_text_of_a_post_emptied_later(self):
        batches = [[_post(1, "First")], [_post(2, "Second")], [_post(3, "Third")]]
        reader, _navigate, _resolve = _reader(_Feed(batches, mounted=1, emptied=True))

        result = await reader.get_my_posts()

        assert [post["text"] for post in result["posts"]] == [
            "First",
            "Second",
            "Third",
        ]

    async def test_only_posts_within_since_days(self):
        # Out of page order on purpose: the result is ordered by publish time.
        recent, older, old = _post(1), _post(5), _post(31)
        reader, _navigate, _resolve = _reader(_Feed([[older, recent, old]]))

        result = await reader.get_my_posts(since_days=30)

        assert _urns(result) == [recent["urn"], older["urn"]]
        assert "section_errors" not in result


class TestTheFeedRead:
    async def test_the_users_own_activity_after_resolving_the_username(self):
        reader, navigate, resolve = _reader(_Feed([[_post(1)]]))

        result = await reader.get_my_posts()

        resolve.assert_awaited_once()
        expected = "https://www.linkedin.com/in/ada-lovelace/recent-activity/all/"
        navigate.assert_awaited_once_with(expected)
        assert result["url"] == expected

    async def test_a_company_page_as_a_member_sees_it(self):
        reader, navigate, resolve = _reader(_Feed([[_post(1)]]))

        await reader.get_my_posts("https://www.linkedin.com/company/acme/posts/")

        resolve.assert_not_awaited()
        navigate.assert_awaited_once_with(
            "https://www.linkedin.com/company/acme/posts/?feedView=all&viewAsMember=true"
        )

    async def test_the_result_carries_each_post_with_its_date(self):
        post = _post(1, "Ada Lovelace\nNotes on the engine\n312 impressions")
        reader, _navigate, _resolve = _reader(_Feed([[post]]))

        result = await reader.get_my_posts()

        urn = post["urn"]
        assert result["posts"] == [
            {
                "urn": urn,
                "url": f"/feed/update/{urn}/",
                "published_at": "2026-10-12T12:00:00Z",
                "text": post["text"],
            }
        ]
        assert result["references"] == {
            "posts": [
                {
                    "kind": "feed_post",
                    "url": f"/feed/update/{urn}/",
                    "text": "Ada Lovelace",
                }
            ]
        }
        assert result["sections"]["posts"] == (
            f"--- {urn} | published 2026-10-12T12:00:00Z ---\n{post['text']}"
        )
