#!/usr/bin/env python3
"""Check (and with ``--install`` pip-install) the core dependencies from ``pyproject.toml``.

    .venv/bin/python scripts/check_deps.py            # exit 1 and list what is missing
    .venv/bin/python scripts/check_deps.py --install  # pip install the missing ones (gpu_up.sh ilab, after rsync)

An rsync'd checkout keeps its old ``.venv``: a dependency added since (``qrcode``) killed the remote
server at import. Checks distributions by name, not versions (an older pin still imports).
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def core_deps(pyproject: Path = ROOT / "pyproject.toml") -> list[str]:
    return list(tomllib.loads(pyproject.read_text())["project"]["dependencies"])


def dist_name(req: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", req.strip(), maxsplit=1)[0]


def missing(reqs: list[str]) -> list[str]:
    out = []
    for req in reqs:
        try:
            md.version(dist_name(req))
        except md.PackageNotFoundError:
            out.append(req)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true", help="pip install the missing core dependencies")
    ap.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    args = ap.parse_args(argv)
    need = missing(core_deps(args.pyproject))
    if not need:
        print("deps: all core dependencies installed")
        return 0
    print(f"deps: missing {' '.join(need)}", file=sys.stderr)
    if not args.install:
        return 1
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
