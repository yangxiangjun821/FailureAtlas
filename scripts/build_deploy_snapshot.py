"""Create the lightweight, read-only data snapshot used by public deployments."""

from __future__ import annotations

import json
from pathlib import Path

from failure_atlas.web import build_dashboard_payload


def main() -> None:
    output = Path("deploy/dashboard.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload()
    payload["deployment_mode"] = "showcase"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Deployment snapshot: {output} ({output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
