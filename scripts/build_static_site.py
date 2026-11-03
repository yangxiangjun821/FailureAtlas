"""Build the static Sites-ready FailureAtlas showcase into an isolated project."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = ROOT / "sites" / "failure-atlas-showcase"
DIST = SITE_ROOT / "dist"


def main() -> None:
    (DIST / "static").mkdir(parents=True, exist_ok=True)
    (DIST / "data").mkdir(parents=True, exist_ok=True)
    (DIST / "reports").mkdir(parents=True, exist_ok=True)

    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    marker = '<script src="/static/app.js" defer></script>'
    index = index.replace(
        marker,
        '<script>window.FAILUREATLAS_STATIC = true;</script>\n    ' + marker,
    )
    (DIST / "index.html").write_text(index, encoding="utf-8")
    shutil.copy2(ROOT / "web" / "static" / "styles.css", DIST / "static" / "styles.css")
    shutil.copy2(ROOT / "web" / "static" / "app.js", DIST / "static" / "app.js")
    shutil.copy2(ROOT / "deploy" / "dashboard.json", DIST / "data" / "dashboard.json")
    shutil.copy2(
        ROOT / "reports" / "generated" / "repair_report.md",
        DIST / "reports" / "repair_report.md",
    )
    print(f"Static showcase built at: {DIST}")


if __name__ == "__main__":
    main()
