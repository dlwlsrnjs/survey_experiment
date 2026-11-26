from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from .util import sha256_file


def render_typography(statement: str, output: Path, font_path: Path | None = None) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1024, 1024
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if font_path and font_path.is_file():
        font = ImageFont.truetype(str(font_path), 48)
    else:
        font = ImageFont.load_default(size=48)
    wrapped = textwrap.wrap(statement.strip(), width=32) or [""]
    line_height = 64
    total_height = line_height * len(wrapped)
    y = max(60, (height - total_height) // 2)
    for line in wrapped:
        box = draw.textbbox((0, 0), line, font=font)
        x = max(40, (width - (box[2] - box[0])) // 2)
        draw.text((x, y), line, fill="black", font=font)
        y += line_height
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False)
    return {
        "status": "success", "tool": "deterministic_typography_renderer_v1",
        "path": str(output.resolve()), "sha256": sha256_file(output),
        "width": width, "height": height, "font": str(font_path) if font_path else "Pillow default",
    }

