"""Tests for reading the user's connections and sent invitations."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from patchright.async_api import Page

from linkedin_mcp_server.linkedin import session as session_module
from linkedin_mcp_server.linkedin.content import PageContentReader
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.network import CONNECTIONS_URL, NetworkReader
from linkedin_mcp_server.linkedin.session import PageSession

PAUSE = 3.0


class _List:
    """A profile list that loads one batch per scroll, as LinkedIn's does.

    *sizes* is how many profiles the list holds after each batch. *ignored*
    names scrolls (1-based) that load nothing. Every scroll and every sleep is
    appended to *log*, in order.
    """

    def __init__(self, sizes: list[int], log: list[Any], ignored: Iterable[int] = ()):
        self.sizes = sizes
        self.loaded = 0
        self.scrolls = 0
        self.ignored = set(ignored)
        self.log = log
        self.url = CONNECTIONS_URL

    async def wait_for_function(self, expression: str, *, timeout: int) -> None:
        return None

    async def evaluate(self, expression: str) -> Any:
        if "linksOnlyTo" in expression:
            return []
        if "return slugs.size;" in expression:
            return self.sizes[self.loaded]
        if "element.scrollTop = element.scrollHeight" in expression:
            self.scrolls += 1
            self.log.append("scroll")
            if self.scrolls not in self.ignored and self.loaded + 1 < len(self.sizes):
                self.loaded += 1
            return True
        raise AssertionError(f"unexpected program: {expression[:80]}")


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every sleep, in order with the scrolls; a clock that moves only then."""
    events: list[Any] = []
    clock = SimpleNamespace(now=0.0)

    async def sleep(seconds: float) -> None:
        events.append(("sleep", seconds))
        clock.now += seconds

    monkeypatch.setattr(session_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(session_module.random, "uniform", lambda low, high: high)
    monkeypatch.setattr(session_module, "detect_rate_limit", AsyncMock())
    monkeypatch.setattr(
        session_module, "handle_modal_close", AsyncMock(return_value=False)
    )
    return events


def _reader(page: _List) -> NetworkReader:
    session = PageSession(cast(Page, page), (PAUSE, PAUSE))
    navigator = cast(PageNavigator, SimpleNamespace(_navigate_to_page=AsyncMock()))
    content = cast(
        PageContentReader,
        SimpleNamespace(
            _extract_root_content=AsyncMock(return_value={"text": "Ada Lovelace"})
        ),
    )
    return NetworkReader(session, navigator, content)


class TestPacing:
    async def test_every_scroll_waits_the_full_pause_the_retry_too(
        self, log: list[Any]
    ):
        # One batch, then nothing: a scroll that loads, one that does not, and
        # the retry after it.
        page = _List([10, 20], log)

        await _reader(page).get_connections(limit=100)

        assert page.scrolls == 3
        for index, event in enumerate(log):
            if event == "scroll":
                assert log[index - 1] == ("sleep", PAUSE), log

    async def test_no_scroll_once_the_limit_is_loaded(self, log: list[Any]):
        page = _List([10, 20, 30], log)

        await _reader(page).get_connections(limit=20)

        assert page.scrolls == 1


class TestLoading:
    async def test_a_retry_that_loads_keeps_the_read_going(self, log: list[Any]):
        page = _List([10, 20, 30], log, ignored={1})

        await _reader(page).get_connections(limit=30)

        assert page.loaded == 2
