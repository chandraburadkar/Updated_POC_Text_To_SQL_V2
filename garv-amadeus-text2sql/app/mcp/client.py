# app/mcp/client.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import anyio
from fastmcp import Client


def _extract_json_from_mcp_result(result: Any) -> Dict[str, Any]:
    """
    fastmcp Client returns a list of content blocks.
    Usually: [TextContent(text="...json...")]
    """
    if isinstance(result, list) and result:
        first = result[0]
        text = getattr(first, "text", None)
        if isinstance(text, str) and text.strip():
            return json.loads(text)
    # fallback (already dict)
    if isinstance(result, dict):
        return result
    return {}


async def _call_tool_async(mcp_url: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    async with Client(mcp_url) as c:
        res = await c.call_tool(tool, args)
        return _extract_json_from_mcp_result(res)


def call_tool(mcp_url: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sync wrapper so your current sync pipeline can call MCP.
    """
    return anyio.run(_call_tool_async, mcp_url, tool, args)


def run_sql_mcp(mcp_url: str, sql: str, limit: int = 50) -> Dict[str, Any]:
    return call_tool(mcp_url, "run_sql", {"sql": sql, "limit": int(limit)})