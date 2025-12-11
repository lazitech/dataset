"""
Author : 朱真言 吕安哲
Description:
    用于生成混乱背景主体辨识问答训练集 (dataset2) 的脚本。数据包含一个有色主体和随机噪点及干扰线背景的图像，
    以及相应的问答对和标准答案。
"""
from __future__ import annotations

import argparse
import math
import random
import string
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Tuple, List, Dict, Any

import colorsys
from PIL import Image, ImageChops, ImageDraw, ImageFont


Color = Tuple[int, int, int]


DEFAULT_SIZE = (224, 224)
SHAPE_LABELS = ["triangle", "circle", "star", "square"]
TEXT_LABELS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
ALL_LABELS = TEXT_LABELS + SHAPE_LABELS

CHARTQA_COLORS: Dict[str, str] = {
    "red": "#d62728",
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "cyan": "#17becf",
    "yellow": "#bcbd22",
    "black": "#000000",
    "white": "#ffffff",
    "navy": "#000080",
    "teal": "#008080",
    "maroon": "#800000",
    "olive": "#808000"
}


QUESTIONS = [
    "<image>\nWhat is the color and the object in this image?",
    "<image>\nIdentify the color and the item shown.",
    "<image>\nDescribe the object's color and shape.",
    "<image>\nWhat color is the foreground element?",
    "<image>\nPlease identify the character or shape and its color."
]

def load_font(font_size: int) -> ImageFont.ImageFont:
    """Try common TrueType fonts; fallback to default bitmap font."""
    font_candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)  
    return font.getsize(text)


def hsv_to_rgb_int(h: float, s: float, v: float) -> Color:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


def normalize_color(color: str | Color) -> Color:
    if isinstance(color, tuple):
        return color
    normalized = color.strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if len(normalized) != 6:
        raise ValueError(f"Unsupported color value: {color}")
    return tuple(int(normalized[i : i + 2], 16) for i in (0, 2, 4))


def add_noise(
    draw: ImageDraw.ImageDraw,
    size: Tuple[int, int],
    dot_count: int,
    dot_color: Color | None,
) -> None:
    width, height = size
    for _ in range(dot_count):
        radius = random.randint(1, max(2, width // 40))
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        if dot_color is not None:
            color = (*dot_color, 255)
        else:
            h = random.random()
            s = random.uniform(0.2, 1.0)
            v = random.uniform(0.4, 0.95)
            color = (*hsv_to_rgb_int(h, s, v), 255)
        draw.ellipse(bbox, fill=color)


def add_interference_lines(draw: ImageDraw.ImageDraw, size: Tuple[int, int], base_h: float, base_v: float, count: int) -> None:
    width, height = size
    for _ in range(count):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        w = random.randint(1, 3)
        h = random.random()
        s = random.uniform(0.3, 0.9)
        v = random.uniform(0.35, 0.95)
        color = (*hsv_to_rgb_int(h, s, v), 255)
        draw.line((x1, y1, x2, y2), fill=color, width=w)


def add_speckle(image: Image.Image, intensity: float = 0.15) -> None:
    noise = Image.effect_noise(image.size, 64).convert("L")
    noise_rgba = Image.merge("RGBA", (noise, noise, noise, Image.new("L", image.size, 255)))
    blended = ImageChops.blend(image, noise_rgba, alpha=intensity)
    image.paste(blended)


def draw_triangle(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], color: Color) -> None:
    left, top, right, bottom = bbox
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    radius = min(right - left, bottom - top) / 2
    pts = []
    for i in range(3):
        angle = -math.pi / 2 + i * 2 * math.pi / 3
        pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    draw.polygon(pts, fill=color)


def draw_circle(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], color: Color) -> None:
    draw.ellipse(bbox, fill=color)


def draw_star(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], color: Color) -> None:
    left, top, right, bottom = bbox
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    outer = min(right - left, bottom - top) / 2
    inner = outer * 0.5
    pts = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = outer if i % 2 == 0 else inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=color)


def draw_square(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], color: Color) -> None:
    draw.rectangle(bbox, fill=color)


def draw_text(draw: ImageDraw.ImageDraw, label: str, bbox: Tuple[int, int, int, int], color: Color) -> None:
    left, top, right, bottom = bbox
    target_w = right - left
    target_h = bottom - top
    effective_w = target_w * 0.4
    effective_h = target_h * 0.4
    low, high = 8, max(int(effective_w), int(effective_h)) * 3
    best_font = load_font(low)
    best_w, best_h = measure_text(draw, label, best_font)
    for _ in range(12):
        mid = (low + high) // 2
        font = load_font(mid)
        w, h = measure_text(draw, label, font)
        if w <= effective_w * 0.98 and h <= effective_h * 0.98:
            best_font, best_w, best_h = font, w, h
            low = mid + 1
        else:
            high = mid - 1

    x = left + (target_w - best_w) / 2
    y = top + (target_h - best_h) / 2 - target_h * 0.05
    draw.text((x, y), label, fill=color, font=best_font)


SHAPE_DRAWERS: dict[str, Callable[[ImageDraw.ImageDraw, Tuple[int, int, int, int], Color], None]] = {
    "triangle": draw_triangle,
    "circle": draw_circle,
    "star": draw_star,
    "square": draw_square,
}


def render_label(
    label: str,
    size: Tuple[int, int],
    bg_color: Color | None = None,
    dot_color: Color | None = None,
    fg_color: Color | None = None,
) -> Image.Image:
    bg_h = random.random()
    bg_v = random.uniform(0.65, 0.95)
    bg_s = 0.25
    bg = bg_color if bg_color is not None else hsv_to_rgb_int(bg_h, bg_s, bg_v)
    
    img = Image.new("RGBA", size, (*bg, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    
    add_noise(
        draw,
        size,
        dot_count=max(200, size[0] * size[1] // 30),
        dot_color=dot_color,
    )
    add_interference_lines(draw, size, base_h=bg_h, base_v=bg_v, count=max(5, min(20, size[0] // 20)))
    add_speckle(img, intensity=0.15)

    if fg_color is not None:
        fg = fg_color
    else:
        fg_h = (bg_h + random.uniform(-0.1, 0.4)) % 1.0
        fg_s = random.uniform(0.5, 1.0)
        fg_v = random.uniform(0.3, 0.95)
        fg = hsv_to_rgb_int(fg_h, fg_s, fg_v)
        
    if label in SHAPE_DRAWERS:
        margin = int(min(size) * 0.35)
    else:
        margin = int(min(size) * 0.08)
    bbox = (margin, margin, size[0] - margin, size[1] - margin)
    fg_alpha = random.randint(220, 255)

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    
    if label in SHAPE_DRAWERS:
        SHAPE_DRAWERS[label](overlay_draw, bbox, (*fg, fg_alpha))
    else:
        draw_text(overlay_draw, label, bbox, (*fg, fg_alpha))
    
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def generate_cot_answer(label: str, color_name: str) -> str:
    """Generate a standard answer for reference in the JSON."""
    if label in SHAPE_LABELS:
        obj_desc = f"{label}"
    elif label.isdigit():
        obj_desc = f"number {label}"
    elif label.isalpha():
        obj_desc = f"character '{label}'"
    else:
        obj_desc = label

    templates = [
        f"The object is a {color_name} {obj_desc}.",
        f"It is a {color_name} {obj_desc}.",
        f"{color_name} {obj_desc}."
    ]
    return random.choice(templates)


def generate_test_dataset(
    labels: Iterable[str],
    samples_per_label: int,
    out_dir: Path,
    image_size: Tuple[int, int] = DEFAULT_SIZE,
    seed: int = 42, 
) -> None:
    random.seed(seed)

    images_dir = out_dir / "test_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    json_data: List[Dict[str, Any]] = []

    def prefix_for(label: str) -> str:
        if label in SHAPE_LABELS:
            return "shape"
        if label in string.digits:
            return "digit"
        if label.isalpha():
            return "char_upper" if label.isupper() else "char_lower"
        return "char"

    counters: dict[str, int] = {}
    for label in labels:
        counters[label] = 0

    total_images = len(list(labels)) * samples_per_label
    print(f"Generating {total_images} TEST images to {out_dir}...")

    color_keys = list(CHARTQA_COLORS.keys())

    for label in labels:
        for _ in range(samples_per_label):
            idx = counters[label] + 1
            counters[label] = idx
            
            chosen_color_name = random.choice(color_keys)
            chosen_rgb = normalize_color(CHARTQA_COLORS[chosen_color_name])
            
            img = render_label(
                label, 
                image_size, 
                bg_color=None, 
                dot_color=None, 
                fg_color=chosen_rgb
            )
            
            fname = f"test_{prefix_for(label)}_{label}_{idx:03d}.png"
            img.save(images_dir / fname)
            
            unique_id = f"test_{prefix_for(label)}_{label}_{idx:03d}"
            question = random.choice(QUESTIONS)
            answer = generate_cot_answer(label, chosen_color_name)
            
            entry = {
                "id": unique_id,
                "image": f"test_images/{fname}",
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
                "gt_info": {
                    "color": chosen_color_name,
                    "label": label,
                    "category": prefix_for(label)
                }
            }
            json_data.append(entry)

    json_path = out_dir / "test_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Test Set generation complete.")
    print(f"Images: {images_dir}")
    print(f"Annotation (with GT): {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ChartQA-colored TEST dataset.")
    parser.add_argument("--out", type=Path, default=Path("dataset_test"), help="Output directory root")
    parser.add_argument("--per-label", type=int, default=2, help="Images per label (Default small for testing)")
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=DEFAULT_SIZE, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (Fixed for reproducibility)")
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(ALL_LABELS),
        help="Comma-separated labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [lbl for lbl in args.labels.split(",") if lbl]
    generate_test_dataset(
        labels,
        args.per_label,
        args.out,
        tuple(args.size),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()