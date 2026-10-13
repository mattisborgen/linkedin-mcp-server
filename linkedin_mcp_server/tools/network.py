"""
LinkedIn network reading tools.

Read the authenticated user's own connections and pending sent invitations.
Both are read-only; withdrawing an invitation is not offered.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error

logger = logging.getLogger(__name__)


def register_network_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register network-list tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Connections",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network"},
    )
    async def get_connections(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=500)] = 50,
    ) -> dict[str, Any]:
        """
        List the authenticated user's most recently added connections, newest first.

        Args:
            ctx: FastMCP context for progress reporting
            limit: How many connections to load (1-500, default 50). The list
                loads in batches, so the text may hold a few more.

        Returns:
            Dict with url, sections["connections"] (raw text: each connection's
            name, headline and the date they connected, as LinkedIn shows it),
            and references["connections"]: up to ``limit`` entries of
            {kind: "person", url: "/in/<username>/", text}, in page order.
            Use the dates in the text to decide which connections are new.
        """
        try:
            extractor = await get_ready_extractor(ctx, tool_name="get_connections")
            logger.info("Reading connections (limit=%d)", limit)
            await ctx.report_progress(
                progress=0, total=100, message="Reading connections"
            )
            result = await extractor.get_connections(limit)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_connections")
        except Exception as e:
            raise_tool_error(e, "get_connections")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Sent Invitations",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network"},
    )
    async def get_sent_invitations(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        """
        List the authenticated user's pending sent connection invitations, newest first.

        Args:
            ctx: FastMCP context for progress reporting
            limit: How many invitations to load (1-500, default 100). The list
                loads in batches, so the text may hold a few more.

        Returns:
            Dict with url, sections["sent_invitations"] (raw text: each
            invitee's name, headline and how long ago the invitation was sent,
            as LinkedIn shows it), and references["sent_invitations"]: up to
            ``limit`` entries of {kind: "person", url: "/in/<username>/", text},
            in page order. Accepted, withdrawn and expired invitations are not
            listed.
        """
        try:
            extractor = await get_ready_extractor(ctx, tool_name="get_sent_invitations")
            logger.info("Reading sent invitations (limit=%d)", limit)
            await ctx.report_progress(
                progress=0, total=100, message="Reading sent invitations"
            )
            result = await extractor.get_sent_invitations(limit)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_sent_invitations")
        except Exception as e:
            raise_tool_error(e, "get_sent_invitations")  # NoReturn
