#!/usr/bin/env python3
"""Smoke-test W&B MCP auth (hosted endpoint, same path as .cursor/scripts/wandb-mcp.sh)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

MCP_URL = "https://mcp.withwandb.com/mcp"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    api_key = os.getenv("WANDB_API_KEY", "").strip()
    if not api_key:
        print("FAIL: WANDB_API_KEY missing in .env", file=sys.stderr)
        return 1

    payload = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    ).encode()
    request = urllib.request.Request(
        MCP_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read(512).decode("utf-8", errors="replace")
        print(f"FAIL: HTTP {exc.code} from hosted MCP: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"FAIL: could not reach hosted MCP: {exc}", file=sys.stderr)
        return 1

    if "query_wandb_tool" not in body and "tools" not in body.lower():
        print(f"FAIL: unexpected hosted MCP response: {body[:300]}", file=sys.stderr)
        return 1

    print("OK: hosted W&B MCP auth works (tools/list succeeded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
