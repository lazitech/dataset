"""
Author : 朱真言 吕安哲
Description:
    用于生成双颜色问答训练集 (dataset1) 的脚本。生成的训练集包含各种背景色和前景色组合的图像，
    以及相应的问答对。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import string
import shutil
from pathlib import Path
from typing import Callable, Iterable, Tuple, Dict, List, Any

import colorsys
from PIL import Image, ImageDraw, ImageFont


Color = Tuple[int, int, int]

DEFAULT_SIZE = (224, 224)
SHAPE_LABELS = ["triangle", "circle", "star", "square"]
TEXT_LABELS = list(string.digits + string.ascii_uppercase)
ALL_SHAPES = TEXT_LABELS + SHAPE_LABELS

CHART_COLORS: Dict[str, str] = {
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

def parse_color_hex(hex_value: str) -> Color:
    """Parse #RRGGBB to tuple."""
    hex_value = hex_value.strip().lstrip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4)) # type: ignore

COLOR_MAP_RGB = {k: parse_color_hex(v) for k, v in CHART_COLORS.items()}
COLOR_KEYS = list(CHART_COLORS.keys())

PROMPTS_TYPE_1 = [
    "The background is {bg_color}. What is the color of the {obj}?",
    "On a {bg_color} background, identify the color of the center {obj}.",
    "Seeing that the background is {bg_color}, what color is the {obj}?",
    "Find the {obj} on the {bg_color} surface and tell me its color.",
]

PROMPTS_TYPE_2 = [
    "There is a {fg_color} {obj}. What is the color of the background?",
    "The {obj} is {fg_color}. What color lies behind it?",
    "Identify the background color behind the {fg_color} {obj}.",
]

PROMPTS_TYPE_3 = [
    "Describe the colors of the object and the background in this image.",
    "What are the colors of the {obj} and the background?",
    "Analyze the color contrast in the image.",
    "Please declare the color of the background and the color of the subject."
]

def load_font(font_size: int) -> ImageFont.ImageFont:
    font_candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "Verdana.ttf"]
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
    return font.getsize(text) # type: ignore

def add_smart_noise(
    draw: ImageDraw.ImageDraw,
    size: Tuple[int, int],
    bg_color: Color,
    dot_count: int,
) -> None:
    """
    Adds noise that is 'tone-on-tone' with the background.
    Instead of random colors, use slightly lighter or darker versions of the bg.
    """
    width, height = size
    bg_r, bg_g, bg_b = bg_color
    
    for _ in range(dot_count):
        radius = random.randint(1, max(2, width // 50))
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        
        delta = random.randint(-30, 30)
        nr = max(0, min(255, bg_r + delta))
        ng = max(0, min(255, bg_g + delta))
        nb = max(0, min(255, bg_b + delta))
        
        draw.ellipse(bbox, fill=(nr, ng, nb))

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
    effective_w = target_w * 0.7
    effective_h = target_h * 0.7
    
    low, high = 10, int(target_h)
    best_font = load_font(low)
    best_w, best_h = measure_text(draw, label, best_font)
    
    for _ in range(10):
        mid = (low + high) // 2
        font = load_font(mid)
        w, h = measure_text(draw, label, font)
        if w <= effective_w and h <= effective_h:
            best_font, best_w, best_h = font, w, h
            low = mid + 1
        else:
            high = mid - 1

    x = left + (target_w - best_w) / 2
    y = top + (target_h - best_h) / 2 - target_h * 0.05
    draw.text((x, y), label, fill=color, font=best_font)

SHAPE_DRAWERS: dict[str, Callable] = {
    "triangle": draw_triangle,
    "circle": draw_circle,
    "star": draw_star,
    "square": draw_square,
}

def render_sample(
    label: str,
    bg_name: str,
    fg_name: str,
    size: Tuple[int, int],
) -> Image.Image:
    
    bg_rgb = COLOR_MAP_RGB[bg_name]
    fg_rgb = COLOR_MAP_RGB[fg_name]
    
    img = Image.new("RGB", size, bg_rgb)
    draw = ImageDraw.Draw(img)
    add_smart_noise(draw, size, bg_rgb, dot_count=random.randint(50, 150))

    if label in SHAPE_DRAWERS:
        margin = int(min(size) * 0.35)
    else:
        margin = int(min(size) * 0.15)
    bbox = (margin, margin, size[0] - margin, size[1] - margin)

    if label in SHAPE_DRAWERS:
        SHAPE_DRAWERS[label](draw, bbox, fg_rgb)
    else:
        draw_text(draw, label, bbox, fg_rgb)
        
    return img

def generate_conversation(label: str, bg_color: str, fg_color: str) -> Tuple[str, str]:
    
    obj_name = label if len(label) > 1 else f"character '{label}'"
    
    mode = random.choice(["ask_fg", "ask_bg", "describe_all"])
    
    if mode == "ask_fg":
        tmpl = random.choice(PROMPTS_TYPE_1)
        question = tmpl.format(bg_color=bg_color, obj=obj_name)
        answer = fg_color
        
    elif mode == "ask_bg":
        tmpl = random.choice(PROMPTS_TYPE_2)
        question = tmpl.format(fg_color=fg_color, obj=obj_name)
        answer = bg_color
        
    else: 
        tmpl = random.choice(PROMPTS_TYPE_3)
        question = tmpl.format(obj=obj_name)
        answer_formats = [
            f"The image shows a {fg_color} {obj_name} on a {bg_color} background.",
            f"Background: {bg_color}. Subject: {fg_color}.",
            f"It is a {fg_color} {obj_name} and the background is {bg_color}."
        ]
        answer = random.choice(answer_formats)
        
    return question, answer

def generate_dataset(
    out_dir: Path,
    samples_total: int,
    image_size: Tuple[int, int] = DEFAULT_SIZE,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset_json: List[Dict[str, Any]] = []
    
    print(f"Generating {samples_total} dual-color samples...")

    for i in range(samples_total):
        label = random.choice(ALL_SHAPES)
        
        bg_name = random.choice(COLOR_KEYS)
        fg_name = random.choice(COLOR_KEYS)
        
        while bg_name == fg_name:
            fg_name = random.choice(COLOR_KEYS)

        image_id = f"{i:06d}"
        filename = f"{image_id}.png"
        
        img = render_sample(label, bg_name, fg_name, image_size)
        img.save(images_dir / filename)
        
        question, answer = generate_conversation(label, bg_name, fg_name)
        
        entry = {
            "id": image_id,
            "image": str(Path("images") / filename),
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        dataset_json.append(entry)

    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
        
    print(f"Done. Dataset saved to {out_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Dual-Color Fine-tuning Dataset Generator")
    parser.add_argument("--out", type=Path, default=Path("dataset_dual_color"), help="Output directory")
    parser.add_argument("--count", type=int, default=1000, help="Number of samples")
    parser.add_argument("--size", type=int, nargs=2, default=DEFAULT_SIZE, help="Image size (W H)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    generate_dataset(args.out, args.count, tuple(args.size), seed=args.seed)

if __name__ == "__main__":
    main()