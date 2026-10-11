"""Render docs/STAGE_CARD.md to a one-page PDF (T-175) with headless Chrome.

    .venv/bin/python scripts/stage_card_pdf.py [--out docs/STAGE_CARD.pdf]

Tries the largest type first and shrinks until the card fits one US Letter page; fails if it never does.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "STAGE_CARD.md"
OUT = ROOT / "docs" / "STAGE_CARD.pdf"
CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
]
SIZES_PT = [x / 4 for x in range(64, 39, -1)]  # 16 pt down to 10 pt

CSS = """
@page { size: letter landscape; margin: 0.25in 0.3in; }
* { box-sizing: border-box; }
body { font: %(pt)spt/1.17 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #000; margin: 0;
       column-count: 2; column-gap: 0.3in; column-rule: 1px solid #999; }
h1 { column-span: all; font-size: 1.35em; margin: 0 0 0.15em; border-bottom: 3px solid #000; }
h1 + p { column-span: all; margin: 0 0 0.4em; padding: 0.15em 0.4em; border: 2px solid #000; }
h2 { font-size: 1.05em; margin: 0.35em 0 0.1em; background: #000; color: #fff; padding: 0.05em 0.3em; }
ul, ol { margin: 0.1em 0; padding-left: 1.3em; }
li { margin: 0.08em 0; break-inside: avoid; }
p { margin: 0.15em 0; }
table { border-collapse: collapse; width: 100%%; }
td, th { border: 1px solid #000; padding: 0.05em 0.3em; text-align: left; vertical-align: top; }
th { background: #ddd; }
code { font-family: ui-monospace, Menlo, monospace; font-size: 0.92em; }
"""


def find_chrome() -> str:
    for c in CHROME:
        path = c if Path(c).exists() else shutil.which(c)
        if path:
            return path
    sys.exit("stage_card_pdf: no Chrome/Chromium found")


def page_count(pdf: Path) -> int:
    if shutil.which("pdfinfo"):
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
        return int(re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE).group(1))
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def render(html: str, pdf: Path, chrome: str, tmp: Path) -> None:
    page = tmp / "card.html"
    page.write_text(html, encoding="utf-8")
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={tmp / 'profile'}",
         f"--print-to-pdf={pdf}", page.as_uri()],
        check=True, capture_output=True, timeout=60,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    body = MarkdownIt("commonmark").enable("table").render(args.src.read_text(encoding="utf-8"))
    chrome = find_chrome()
    with tempfile.TemporaryDirectory(prefix="stage-card-") as d:
        tmp = Path(d)
        for pt in SIZES_PT:
            html = f'<!doctype html><meta charset="utf-8"><style>{CSS % {"pt": pt}}</style>{body}'
            render(html, args.out, chrome, tmp)
            if page_count(args.out) == 1:
                print(f"stage_card_pdf: {args.out} (1 page, {pt} pt)")
                return 0
    print(f"stage_card_pdf: does not fit one page even at {SIZES_PT[-1]} pt", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
