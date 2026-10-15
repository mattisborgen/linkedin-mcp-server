"""The local macOS window guard in ``conftest.py``.

Local runs on macOS keep every browser off the developer's screen; CI keeps the
real modes. Which one a run is comes from ``CI``, read as a boolean.
"""

from __future__ import annotations

import os
import sys

import pytest
from conftest import running_in_ci
from patchright.async_api import BrowserType

from linkedin_mcp_server.hidden_target import CHROMIUM_HEADLESS_ENV


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        (" yes ", True),
    ],
)
def test_ci_is_read_as_a_boolean_and_not_as_presence(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
):
    if value is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", value)

    assert running_in_ci() is expected


@pytest.mark.skipif(
    sys.platform != "darwin" or running_in_ci(),
    reason="the guard covers local macOS runs only",
)
class TestALocalMacosRun:
    def test_uses_chromium_headless_in_every_process_it_starts(self):
        assert os.environ.get(CHROMIUM_HEADLESS_ENV) == "1"

    @pytest.mark.parametrize("launch", ["launch", "launch_persistent_context"])
    async def test_refuses_a_visible_window(self, launch: str):
        # Refused before Patchright is reached, so no browser starts here.
        with pytest.raises(RuntimeError, match="visible browser window"):
            await getattr(BrowserType, launch)(object(), headless=False)
