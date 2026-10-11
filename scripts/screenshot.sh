#!/usr/bin/env bash
# Screenshot a page with the one pinned Playwright: scripts/screenshot.sh <url> <width> <out.png> [height]
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <url> <width> <out.png> [height]" >&2; exit 2; }
cd "$(dirname "$0")/.."
. scripts/_py.sh
"$HEARING_PLAYWRIGHT" - "$1" "$2" "$3" "${4:-844}" <<'PY'
import sys
from playwright.sync_api import sync_playwright

url, width, out, height = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": width, "height": height})
    page.goto(url, wait_until="load")
    page.wait_for_timeout(500)
    page.screenshot(path=out, full_page=True)
    browser.close()
print(out)
PY
