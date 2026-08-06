"""Stage the WOFFL web app for a side-instance Databricks Apps deployment.

NOTE: production now deploys the web app straight from the repo pull (root
app.yaml runs uvicorn; web/dist is committed). This script remains for
standing up a TEST instance beside prod via the Databricks CLI.

Builds the SPA, then assembles a minimal deployable tree under
build/webapp_stage/ containing only what the web app needs:

    app.yaml            (copied from the repo root app.yaml)
    requirements.txt    (server deps; streamlit not required)
    server/             FastAPI backend
    woffl/              physics + clients + jp_data (surveys, csv)
    web/dist/           built SPA
    data/               bundled fallbacks (jp history xlsx, pump dims json)

Usage (from repo root):
    python scripts/stage_web_app.py [--skip-build]

Then deploy with the Databricks CLI (your auth, your test app name):
    databricks sync ./build/webapp_stage /Workspace/Users/<you>/woffl-web-test --full
    databricks apps deploy woffl-web-test --source-code-path /Workspace/Users/<you>/woffl-web-test
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / "build" / "webapp_stage"

# Server-side requirements for the deployed web app. FastAPI/uvicorn ship in
# the Databricks Apps default environment but are pinned here for parity with
# local dev. Streamlit is intentionally absent - the server never imports it.
WEB_REQUIREMENTS = """\
fastapi>=0.115
uvicorn>=0.30
matplotlib>=3.8.2
numpy>=1.26.3
pandas>=2.1.4
scipy>=1.11.4
ortools>=9.15
databricks-sql-connector>=3.0.0
databricks-sdk>=0.20.0
python-dotenv>=1.0.0
python-dateutil>=2.8.0
openpyxl>=3.0.0
"""


def run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, shell=sys.platform == "win32")
    if result.returncode != 0:
        sys.exit(result.returncode)


def copy_tree(src: Path, dst: Path, ignore: shutil._IgnorePattern | None = None) -> None:  # type: ignore[name-defined]
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="reuse existing web/dist")
    args = parser.parse_args()

    if not args.skip_build:
        run(["npm", "run", "build"], cwd=REPO / "web")

    dist = REPO / "web" / "dist"
    if not (dist / "index.html").is_file():
        sys.exit("web/dist/index.html missing - build failed or --skip-build without a prior build")

    if STAGE.exists():
        # Windows: copied read-only entries (e.g. .claude worktree dirs from a
        # previous buggy run) make plain rmtree fail - chmod and retry.
        def _chmod_retry(func, path):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(STAGE, onexc=lambda f, p, e: _chmod_retry(f, p))
    STAGE.mkdir(parents=True)

    ignore_py = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", ".claude", ".git*"
    )
    copy_tree(REPO / "server", STAGE / "server", ignore=ignore_py)
    copy_tree(REPO / "woffl", STAGE / "woffl", ignore=ignore_py)
    copy_tree(dist, STAGE / "web" / "dist")
    copy_tree(REPO / "data", STAGE / "data")

    shutil.copy2(REPO / "app.yaml", STAGE / "app.yaml")
    (STAGE / "requirements.txt").write_text(WEB_REQUIREMENTS, encoding="utf-8")

    total_mb = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file()) / 1e6
    print(f"\nStaged {total_mb:.1f} MB at {STAGE}")
    print("Deploy:")
    print("  databricks sync ./build/webapp_stage /Workspace/Users/<you>/woffl-web-test --full")
    print("  databricks apps deploy woffl-web-test --source-code-path /Workspace/Users/<you>/woffl-web-test")


if __name__ == "__main__":
    main()
