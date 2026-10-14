"""Assemble the VoiceLock web demo into deploy/site/ for Vercel.

    .venv/bin/python deploy/build.py          # then: vercel deploy deploy/site --prod

What goes in:
  app/, console/ and the console pages at the root (/who, ...)  from web/, as `hearing serve` mounts them
  app/_files.json, the stamped app/index.html and app/sw.js       what hearing/server/static.py would serve
  demo/shim.js and demo/data/                                     the in-browser stand-in for the server
  demo/media/*.wav                                                 the recorded clips (deploy/media/, not in git)
  api/comfort.py + hearing/{server/comfort*,synth/tts,audio/io,models}.py + config/voices.yaml
                                                                   the Talk to Me function (real Gemini, ElevenLabs)
The demo build also shows About in the nav. Nothing else in the app changes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
SITE = DEPLOY / "site"
sys.path.insert(0, str(ROOT))

from hearing.server.static import app_files, build_stamp, stamp_index, stamp_sw  # noqa: E402

SHIM_TAG = '<script src="/demo/shim.js"></script>'
MEDIA = ("room.wav", "voice.wav", "who.wav")
FUNCTION_FILES = ("hearing/server/comfort.py", "hearing/server/comfort_catalog.json", "hearing/synth/tts.py",
                  "hearing/audio/io.py", "hearing/models.py", "config/voices.yaml")


def inject_shim(html: str) -> str:
    """The shim must run before the app's modules, so it goes first in <head>."""
    if SHIM_TAG in html:
        return html
    return re.sub(r"(<head[^>]*>)", r"\1\n  " + SHIM_TAG, html, count=1)


def main() -> None:
    missing = [m for m in MEDIA if not (DEPLOY / "media" / m).is_file()]
    if missing:
        sys.exit(f"deploy/media is missing {', '.join(missing)} (the recorded clips are kept out of git)")
    link = SITE / ".vercel"  # the Vercel project link survives a rebuild
    kept = None
    if link.is_dir():
        kept = DEPLOY / ".vercel-link"
        shutil.rmtree(kept, ignore_errors=True)
        shutil.move(str(link), kept)
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    if kept:
        shutil.move(str(kept), link)
    ignore = shutil.ignore_patterns("tests", "__pycache__", ".DS_Store", "comp", "*.md")

    # the console mount serves web/console at / (so /who is who.html) and the app imports ../console/*.js
    shutil.copytree(ROOT / "web/console", SITE, ignore=ignore, dirs_exist_ok=True)
    shutil.copytree(ROOT / "web/console", SITE / "console", ignore=ignore)
    shutil.copytree(ROOT / "web/app", SITE / "app", ignore=ignore)

    app_dir = ROOT / "web/app"
    stamp = build_stamp(app_dir)
    (SITE / "app/_files.json").write_text(json.dumps(app_files(app_dir)))
    (SITE / "app/index.html").write_text(inject_shim(stamp_index((app_dir / "index.html").read_text(), stamp)))
    (SITE / "app/sw.js").write_text(stamp_sw((app_dir / "sw.js").read_text(), stamp))
    for page in SITE.glob("*.html"):
        page.write_text(inject_shim(page.read_text()))

    # About is in the nav on the demo (it was cut from the stage build on judging day)
    app_js = SITE / "app/app.js"
    text = app_js.read_text()
    about = '{ id: "about", title: "About", nav: false }'
    if about not in text:
        sys.exit("app.js no longer has the About entry this build expects")
    app_js.write_text(text.replace(about, '{ id: "about", title: "About", nav: true }'))

    shutil.copytree(DEPLOY / "demo", SITE / "demo", ignore=shutil.ignore_patterns(".DS_Store"))
    (SITE / "demo/media").mkdir()
    for m in MEDIA:
        shutil.copy2(DEPLOY / "media" / m, SITE / "demo/media" / m)

    (SITE / "api").mkdir()
    shutil.copy2(DEPLOY / "api/comfort.py", SITE / "api/comfort.py")
    for rel in FUNCTION_FILES:
        (SITE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, SITE / rel)
    shutil.copy2(DEPLOY / "requirements.txt", SITE / "requirements.txt")
    shutil.copy2(DEPLOY / "vercel.json", SITE / "vercel.json")
    shutil.copy2(DEPLOY / ".vercelignore", SITE / ".vercelignore")

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"built {SITE.relative_to(ROOT)} ({size / 1e6:.1f} MB, stamp {stamp})")


if __name__ == "__main__":
    main()
