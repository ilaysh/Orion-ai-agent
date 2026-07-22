# orion_core/brain/mcp_manager.py
import os
import json
from typing import Dict, Optional
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

class MCPManager:
    """Isolated Singleton to maintain live MCP streams outside of LangGraph State."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPManager, cls).__new__(cls)
            cls._instance._sessions: Dict[str, ClientSession] = {}
            cls._instance._stacks: Dict[str, AsyncExitStack] = {}
            cls._instance._config_cache: Optional[Dict] = None
        return cls._instance

    def _get_config(self) -> Dict:
        if self._config_cache is None:
            config_path = "orion_core/memory/mcp_config.json"
            if not os.path.exists(config_path): return {}
            with open(config_path, "r") as f:
                raw = json.load(f)
                self._config_cache = raw.get("mcpServers", raw)
        return self._config_cache

    async def get_session(self, server_name: str) -> ClientSession:
        if server_name in self._sessions:
            return self._sessions[server_name]
            
        config = self._get_config().get(server_name)
        if not config: 
            raise ValueError(f"MCP Server '{server_name}' not found in mcp_config.json.")

        env = os.environ.copy()
        env.update(config.get("env", {}))
        params = StdioServerParameters(command=config["command"], args=config.get("args", []), env=env)

        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        self._sessions[server_name] = session
        self._stacks[server_name] = stack
        return session

    def _search_server_name(self) -> Optional[str]:
        """Which configured MCP server handles web search. NO hardcoded provider
        name (that was the 'brave-search' bug). Preference order: a server whose
        config sets "role": "search", then a server literally named 'search', then
        the first/only server in the config. The config is the single source of
        truth — swapping search providers is a config edit, not a code change."""
        cfg = self._get_config()
        if not cfg:
            return None
        for name, spec in cfg.items():
            if isinstance(spec, dict) and spec.get("role") == "search":
                return name
        if "search" in cfg:
            return "search"
        return next(iter(cfg), None)

    async def call_search(self, query: str) -> str:
        """Run a web search via the configured search server, discovering that
        server's actual search-tool NAME at runtime (Tavily, Brave, etc. all name
        it differently) so nothing is hardcoded."""
        name = self._search_server_name()
        if not name:
            raise ValueError("No search server is configured in mcp_config.json.")
        session = await self.get_session(name)

        listed = await session.list_tools()
        tools = getattr(listed, "tools", listed) or []
        tool_name = next((t.name for t in tools if "search" in t.name.lower()), None)
        if tool_name is None and tools:
            tool_name = tools[0].name
        if tool_name is None:
            raise ValueError(f"MCP server '{name}' exposes no tools.")

        result = await session.call_tool(tool_name, {"query": query})
        return "\n".join(c.text for c in result.content
                         if getattr(c, "type", "") == "text")

# Global instance for tool bindings
mcp_manager = MCPManager()