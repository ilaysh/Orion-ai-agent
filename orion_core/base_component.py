# orion_core/base_component.py
from __future__ import annotations

import asyncio
from enum import Enum, auto
from typing import Awaitable, Callable, Optional, Dict, Any


class ComponentState(Enum):
    STOPPED = auto()
    INITIALIZING = auto()
    READY = auto()
    ERROR = auto()


ReportFn = Callable[[str, bool, Dict[str, Any]], Awaitable[None]]


class BaseComponent:
    """
    Minimal lifecycle interface all components share.
    Core will:
      - set reporter via set_reporter(...)
      - call init() once at boot
      - call start()/stop() to enable/disable at runtime
    Each component should:
      - set self.state appropriately
      - call self._report(...) after meaningful changes
    """
    name: str = "component"

    def __init__(self) -> None:
        self.state: ComponentState = ComponentState.STOPPED
        self.active: bool = False
        self._report: Optional[ReportFn] = None
        self._lock = asyncio.Lock()

    def set_reporter(self, reporter: ReportFn) -> None:
        """Core injects an async reporter to broadcast readiness to the UI."""
        self._report = reporter

    async def init(self) -> None:
        """Load heavy resources (models, devices). Override in subclass."""
        async with self._lock:
            self.state = ComponentState.INITIALIZING
            await self._report_state()
            # subclass work here
            self.state = ComponentState.READY
            await self._report_state()

    async def start(self) -> None:
        """Begin running (threads/loops). Override in subclass if needed."""
        async with self._lock:
            if self.state is not ComponentState.READY:
                await self.init()
            self.active = True
            await self._report_state(extra={"active": True})

    async def stop(self) -> None:
        """Stop safely. Override in subclass to release resources."""
        async with self._lock:
            self.active = False
            # keep READY so we can re-start quickly; set STOPPED if you tear down
            await self._report_state(extra={"active": False})

    def is_ready(self) -> bool:
        return self.state == ComponentState.READY

    async def _report_state(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if self._report is not None:
            await self._report(self.name, self.is_ready(), extra or {})
