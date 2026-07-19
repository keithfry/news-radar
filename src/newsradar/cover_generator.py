"""Generate podcast cover images (Style A: Tech Dark) using Pillow.

Channel cover: static 1400x1400 PNG per topic, generated once.
Episode cover: per-episode 1400x1400 JPEG with tagline, date, duration.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from .topics import Topic

SIZE = 1400
_BAR_H = 195  # bottom bar height (episode covers)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    import matplotlib
    fonts_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    path = fonts_dir / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def _make_base_image(topic: Topic):
    from PIL import Image

    c_top = np.array(topic.bg_top_color, dtype=float)
    c_mid = np.array(topic.bg_mid_color, dtype=float)

    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for y in range(SIZE):
        t = y / SIZE
        if t < 0.55:
            t2 = t / 0.55
            row = c_top * (1 - t2) + c_mid * t2
        else:
            t2 = (t - 0.55) / 0.45
            row = c_mid * (1 - t2) + c_top * t2
        pixels[y] = row.astype(np.uint8)

    # Radial glow at upper-right
    accent = np.array(topic.accent_color, dtype=float)
    cx, cy = int(SIZE * 0.72), int(SIZE * 0.22)
    radius = SIZE * 0.52

    y_g, x_g = np.mgrid[0:SIZE, 0:SIZE]
    dist = np.sqrt((x_g - cx) ** 2 + (y_g - cy) ** 2)
    alpha = np.clip(1.0 - dist / radius, 0, 1) ** 2.8 * 0.22
    for c in range(3):
        pixels[:, :, c] = np.clip(
            pixels[:, :, c] * (1 - alpha) + accent[c] * alpha, 0, 255
        ).astype(np.uint8)

    return Image.fromarray(pixels, "RGB")


def _center_text(draw, y: int, text: str, font, fill) -> int:
    """Draw text centered horizontally; return bottom y."""
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text(((SIZE - w) // 2, y), text, font=font, fill=fill)
    return y + (bbox[3] - bbox[1])


def _wrap_lines(draw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def generate_channel_cover(topic: Topic, author_name: str, out_path: Path) -> None:
    """Static channel cover for podcast app listing (PNG)."""
    from PIL import ImageDraw

    img = _make_base_image(topic)
    draw = ImageDraw.Draw(img)
    accent = topic.accent_color
    light = topic.accent_color_light

    label = f"{topic.display_name.upper()} DAILY RADAR"
    title_lines = (topic.display_name, "Daily", "Radar")
    byline = f"Daily {topic.display_name} news digest for your listening pleasure"

    # Eyebrow label
    f_eye = _font(45, bold=True)
    _center_text(draw, 170, label, f_eye, accent)

    # Accent rule
    rx = SIZE // 2 - 44
    draw.rectangle([(rx, 238), (rx + 88, 245)], fill=accent)

    # Measure title block height to center it vertically
    f_title = _font(177, bold=True)
    f_sub = _font(45)
    line_heights = [draw.textbbox((0, 0), ln, font=f_title)[3] for ln in title_lines]
    title_block_h = sum(h + 8 for h in line_heights) - 8
    sub_h = draw.textbbox((0, 0), byline, font=f_sub)[3]
    gap = 36
    total_block_h = title_block_h + gap + sub_h

    zone_top = 265       # below accent rule
    zone_bot = SIZE - 160  # above author line
    y = zone_top + (zone_bot - zone_top - total_block_h) // 2

    # Main title
    for line, lh in zip(title_lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=f_title)
        w = bbox[2] - bbox[0]
        draw.text(((SIZE - w) // 2, y), line, font=f_title, fill=(248, 250, 252))
        y += lh + 8

    # Subtitle
    _center_text(draw, y + gap - 8, byline, f_sub, (148, 163, 184))

    # Author
    f_auth = _font(42, bold=True)
    _center_text(draw, SIZE - 130, author_name, f_auth, light)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "PNG", optimize=True)


def generate_episode_cover(
    topic: Topic,
    tagline: str,
    date: datetime,
    duration_sec: int,
    out_path: Path,
) -> None:
    """Per-episode cover with tagline, date, and duration (JPEG)."""
    from PIL import Image, ImageDraw

    img = _make_base_image(topic)
    draw = ImageDraw.Draw(img)
    accent = topic.accent_color
    light = topic.accent_color_light

    label = f"{topic.display_name.upper()} DAILY RADAR"

    # Eyebrow
    f_eye = _font(84, bold=True)
    _center_text(draw, 100, label, f_eye, accent)

    # Accent rule
    rx = SIZE // 2 - 44
    draw.rectangle([(rx, 218), (rx + 88, 225)], fill=accent)

    # Tagline (wrapped, vertically centered in upper zone)
    f_tag = _font(111, bold=True)
    pad = 110
    lines = _wrap_lines(draw, tagline, f_tag, SIZE - pad * 2)
    line_h = draw.textbbox((0, 0), "Ag", font=f_tag)[3] + 18
    zone_top, zone_bot = 230, SIZE - _BAR_H - 40
    total_h = len(lines) * line_h - 18
    y = zone_top + (zone_bot - zone_top - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=f_tag)
        w = bbox[2] - bbox[0]
        draw.text(((SIZE - w) // 2, y), line, font=f_tag, fill=(248, 250, 252))
        y += line_h

    # Bottom bar overlay
    bar_top = SIZE - _BAR_H
    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, bar_top), (SIZE, SIZE)], fill=(*accent, 52))
    od.line([(0, bar_top), (SIZE, bar_top)], fill=(*accent, 80), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Date (centered)
    f_date = _font(51, bold=True)
    date_str = date.strftime("%B %-d, %Y")
    _center_text(draw, bar_top + 20, date_str, f_date, light)

    # Duration (centered)
    f_dur = _font(42)
    mins = duration_sec // 60
    dur_str = f"{mins} minute{'s' if mins != 1 else ''}"
    _center_text(draw, bar_top + 100, dur_str, f_dur, light)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=92, optimize=True)


OG_W, OG_H = 1200, 630
_OG_BAR_H = 185


def _make_og_base_image(topic: Topic):
    """1200×630 dark gradient base for OG social cards."""
    from PIL import Image

    c_top = np.array(topic.bg_top_color, dtype=float)
    c_mid = np.array(topic.bg_mid_color, dtype=float)

    pixels = np.zeros((OG_H, OG_W, 3), dtype=np.uint8)
    for y in range(OG_H):
        t = y / OG_H
        if t < 0.55:
            t2 = t / 0.55
            row = c_top * (1 - t2) + c_mid * t2
        else:
            t2 = (t - 0.55) / 0.45
            row = c_mid * (1 - t2) + c_top * t2
        pixels[y] = row.astype(np.uint8)

    # Radial glow at upper-right
    accent = np.array(topic.accent_color, dtype=float)
    cx, cy = int(OG_W * 0.82), int(OG_H * 0.22)
    radius = OG_W * 0.38

    y_g, x_g = np.mgrid[0:OG_H, 0:OG_W]
    dist = np.sqrt((x_g - cx) ** 2 + (y_g - cy) ** 2)
    alpha = np.clip(1.0 - dist / radius, 0, 1) ** 2.8 * 0.22
    for c in range(3):
        pixels[:, :, c] = np.clip(
            pixels[:, :, c] * (1 - alpha) + accent[c] * alpha, 0, 255
        ).astype(np.uint8)

    return Image.fromarray(pixels, "RGB")


def _og_center_text(draw, y: int, text: str, font, fill) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text(((OG_W - w) // 2, y), text, font=font, fill=fill)
    return y + (bbox[3] - bbox[1])


def generate_og_card(
    topic: Topic,
    tagline: str,
    date: datetime,
    duration_sec: int,
    out_path: Path,
) -> None:
    """1200×630 Open Graph social card for HTML digest pages (JPEG)."""
    from PIL import Image, ImageDraw

    img = _make_og_base_image(topic)
    draw = ImageDraw.Draw(img)
    accent = topic.accent_color
    light = topic.accent_color_light

    label = f"{topic.display_name.upper()} DAILY RADAR"

    # Eyebrow
    f_eye = _font(39, bold=True)
    _og_center_text(draw, 38, label, f_eye, accent)

    # Accent rule
    rx = OG_W // 2 - 44
    draw.rectangle([(rx, 90), (rx + 88, 97)], fill=accent)

    # Tagline — wrap to fit, max 2 lines
    f_tag = _font(108, bold=True)
    pad = 100
    lines = _wrap_lines(draw, tagline, f_tag, OG_W - pad * 2)[:2]
    line_h = draw.textbbox((0, 0), "Ag", font=f_tag)[3] + 14
    zone_top, zone_bot = 112, OG_H - _OG_BAR_H - 10
    total_h = len(lines) * line_h - 14
    y = zone_top + (zone_bot - zone_top - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=f_tag)
        w = bbox[2] - bbox[0]
        draw.text(((OG_W - w) // 2, y), line, font=f_tag, fill=(248, 250, 252))
        y += line_h

    # Bottom bar overlay
    bar_top = OG_H - _OG_BAR_H
    overlay = Image.new("RGBA", (OG_W, OG_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, bar_top), (OG_W, OG_H)], fill=(*accent, 52))
    od.line([(0, bar_top), (OG_W, bar_top)], fill=(*accent, 80), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Date (centered)
    f_date = _font(48, bold=True)
    date_str = date.strftime("%B %-d, %Y")
    _og_center_text(draw, bar_top + 18, date_str, f_date, light)

    # Duration (centered)
    f_dur = _font(39)
    mins = duration_sec // 60
    dur_str = f"{mins} minute{'s' if mins != 1 else ''}"
    _og_center_text(draw, bar_top + 100, dur_str, f_dur, light)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=92, optimize=True)
