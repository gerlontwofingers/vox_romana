"""vox_romana MCP server — expose vox_romana to Copilot/Claude/etc. via MCP.

This package boots a `Model Context Protocol
<https://modelcontextprotocol.io/>`_ server that exposes a curated
slice of vox_romana's read-only surface as MCP tools.  An IDE that
speaks MCP (VS Code Copilot Chat, Claude Desktop, …) can then ask
questions like:

    "What episodes have we made about UBI?"
    "Search the corpus for anything on cooperative game theory."
    "Which lenses are configured, and what do they sound like?"

The server is intentionally read-only in this chunk; write/action
tools (ingest, kick-off-generate, …) are added in a later chunk.

Run it manually with::

    python -m vox_romana.mcp

or wire it into VS Code via ``.vscode/mcp.json`` (see
``.vscode/mcp.json.example`` for a template).
"""

from vox_romana.mcp.server import build_server

__all__ = ["build_server"]
