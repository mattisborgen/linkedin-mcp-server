"""
LinkedIn network reading tools.

Read the authenticated user's own connections, pending sent invitations and
recent posts, or a company page's posts. All are read-only; withdrawing an
invitation is not offered.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.linkedin.activity import normalize_posts_source

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

    @mcp.tool(
        timeout=tool_timeout,
        title="Get My Posts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network"},
    )
    async def get_my_posts(
        ctx: Context,
        source: str = "personal",
        since_days: Annotated[int, Field(ge=1, le=365)] = 30,
        limit: Annotated[int, Field(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        """
        List recent posts and their stats, the user's own or a company page's, newest first.

        Each post's text is what LinkedIn renders: author, body (which may stop
        at "see more") and the counts it shows, such as reactions, comments,
        reposts and, on the user's own posts, impressions. Parse the counts
        from the text. Every entry in posts is {urn, url, published_at, text},
        with published_at the UTC publish time decoded from the URN.

        Args:
            ctx: FastMCP context for progress reporting
            source: "personal" (default) for the authenticated user's own posts,
                or a company's /company/ slug or URL (e.g. "microsoft") for that
                page's posts as a member sees them.
            since_days: Only posts published in the last this many days
                (1-365, default 30).
            limit: Most posts to return (1-200, default 50).

        Returns:
            Dict with url, sections["posts"] (each post's text under a
            "--- <urn> | published <published_at> ---" line),
            references["posts"] ({kind: "feed_post", url: "/feed/update/<urn>/",
            text: first line}) and posts, newest first. section_errors["posts"]
            is set when the page showed no posts at all.
        """
        try:
            # Validate before starting the browser.
            normalize_posts_source(source)
            extractor = await get_ready_extractor(ctx, tool_name="get_my_posts")
            logger.info(
                "Reading posts: %s (since_days=%d, limit=%d)", source, since_days, limit
            )
            await ctx.report_progress(progress=0, total=100, message="Reading posts")
            result = await extractor.get_my_posts(source, since_days, limit)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_my_posts")
        except Exception as e:
            raise_tool_error(e, "get_my_posts")  # NoReturn
