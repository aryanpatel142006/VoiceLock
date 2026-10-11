"""Draw the app icons (T-345): concentric rings of a sound field, one green ring in focus inside a dim amber room.

Run from the repo root: ``.venv/bin/python web/app/icons/make_icons.py``. Pillow only; the PNGs are committed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
BG = (7, 8, 10)
AMBER = (240, 179, 90)
GREEN = (79, 209, 165)
SS = 4  # supersampling


def draw(size: int, *, safe: float = 1.0, rounded: bool = False) -> Image.Image:
    """``safe`` shrinks the mark into the maskable safe zone (0.8 of the canvas)."""
    n = size * SS
    img = Image.new("RGB", (n, n), BG)
    c = n / 2
    r_max = n * 0.40 * safe
    glow = Image.new("RGB", (n, n), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([c - r_max * 0.55, c - r_max * 0.55, c + r_max * 0.55, c + r_max * 0.55], fill=(18, 70, 55))
    glow = glow.filter(ImageFilter.GaussianBlur(n * 0.06))
    img.paste(glow, (0, 0), Image.eval(glow.convert("L"), lambda v: min(255, v * 3)))
    d = ImageDraw.Draw(img)
    for frac, colour, width in ((1.0, AMBER, 0.012), (0.82, AMBER, 0.014), (0.64, AMBER, 0.016)):
        r = r_max * frac
        dim = tuple(int(BG[i] + (colour[i] - BG[i]) * (0.22 + 0.18 * (1 - frac) / 0.36)) for i in range(3))
        d.ellipse([c - r, c - r, c + r, c + r], outline=dim, width=max(1, int(n * width)))
    r = r_max * 0.40
    d.ellipse([c - r, c - r, c + r, c + r], outline=GREEN, width=int(n * 0.034))
    r = r_max * 0.14
    d.ellipse([c - r, c - r, c + r, c + r], fill=(238, 241, 245))
    out = img.resize((size, size), Image.LANCZOS)
    if rounded:
        mask = Image.new("L", (n, n), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.22), fill=255)
        rgba = out.convert("RGBA")
        rgba.putalpha(mask.resize((size, size), Image.LANCZOS))
        return rgba
    return out


def main() -> None:
    draw(192, rounded=True).save(HERE / "icon-192.png", optimize=True)
    draw(512, rounded=True).save(HERE / "icon-512.png", optimize=True)
    draw(512, safe=0.8).save(HERE / "maskable-512.png", optimize=True)
    draw(180).save(HERE / "apple-touch-icon.png", optimize=True)  # iOS rounds it itself; no alpha
    draw(32, rounded=True).save(HERE / "favicon-32.png", optimize=True)


if __name__ == "__main__":
    main()
