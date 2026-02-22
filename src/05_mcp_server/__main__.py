"""Allow ``python -m src.mcp_server`` to launch the server."""

import asyncio
import sys
from pathlib import Path

# Psycopg's async driver requires SelectorEventLoop on Windows.
# Python ≥3.8 defaults to ProactorEventLoop, which is incompatible.
# This MUST run before any asyncio loop is created (i.e. before imports
# that touch the event loop).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mcp_server.server import main

main()
