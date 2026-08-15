"""
MCP-Medimg Server module - re-export from gateway.server.

Provides access to MCP server tools, resources, prompts, and transports.
"""

from gateway.server import *  # noqa: F401,F403
import gateway.server as _gw_server

# Expose internal module-level variables needed for monkeypatching
_gateway = _gw_server._gateway
mcp = _gw_server.mcp
