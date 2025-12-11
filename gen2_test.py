"""
Author: 朱真言 吕安哲   
Description:
    用于生成 dataset2 对应的测试集。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import string
import shutil
import itertools
from pathlib import Path
from typing import Callable, Tuple, Dict, List, Any

from PIL import Image, ImageDraw, ImageFont

# Type alias
Color = Tuple[int, int, int]

DEFAULT_SIZE = (224, 224)
SHAPE_LABELS = ["triangle", "circle", "star", "square"]
TEXT_LABELS = list(string.digits + string.ascii_uppercase)
ALL_SHAPES = TEXT_LABELS + SHAPE_LABELS

# --- ChartQA / Mainstream Colors (Hex) ---
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
    hex_value = hex_value.strip().lstrip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4)) # type: ignore

COLOR_MAP_RGB = {k: parse_color_hex(v) for k, v in CHART_COLORS.items()}
COLOR_KEYS = list(CHART_COLORS.keys())

# --- Prompt Templates ---
# Type 1: Given Background, Identify Foreground
PROMPTS_TYPE_1 = [
    "The background is {bg_color}. What is the color of the {obj}?",
    "On a {bg_color} background, identify the color of the center {obj}.",
    "Seeing that the background is {bg_color}, what color is the {obj}?",
]

# Type 2: Given Foreground, Identify Background
PROMPTS_TYPE_2 = [
    "There is a {fg_color} {obj}. What is the color of the background?",
    "The {obj} is {fg_color}. What color lies behind it?",
    "Identify the background color behind the {fg_color} {obj}.",
]

# Type 3: Describe Both (Simplified for exact match testing)
PROMPTS_TYPE_3 = [
    "What are the colors of the {obj} and the background?",
    "Describe the colors of the object and the background."
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

# --- Drawing Primitives ---
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

def generate_qa_pair(label: str, bg_color: str, fg_color: str) -> Dict[str, Any]:
    """
    Generates structured metadata for evaluation.
    """
    obj_name = label if len(label) > 1 else f"character '{label}'"
    
    mode = random.choice(["ask_fg", "ask_bg", "describe_all"])
    
    if mode == "ask_fg":
        tmpl = random.choice(PROMPTS_TYPE_1)
        question = tmpl.format(bg_color=bg_color, obj=obj_name)
        full_answer = f"The {obj_name} is {fg_color}."
        short_answer = fg_color
        
    elif mode == "ask_bg":
        tmpl = random.choice(PROMPTS_TYPE_2)
        question = tmpl.format(fg_color=fg_color, obj=obj_name)
        full_answer = f"The background is {bg_color}."
        short_answer = bg_color
        
    else: # describe_all
        tmpl = random.choice(PROMPTS_TYPE_3)
        question = tmpl.format(obj=obj_name)
        full_answer = f"Object: {fg_color}. Background: {bg_color}."
        short_answer = f"{fg_color}, {bg_color}" 
        
    return {
        "mode": mode,
        "question": question,
        "full_answer": full_answer,
        "short_answer": short_answer,
    }

def generate_test_set(
    out_dir: Path,
    samples_total: int,
    balanced: bool = True,
    image_size: Tuple[int, int] = DEFAULT_SIZE,
    seed: int = 42
) -> None:
    if seed is not None:
        random.seed(seed)

    images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset_llava: List[Dict[str, Any]] = []
    metadata_list: List[Dict[str, Any]] = []
    
    print(f"Generating {'balanced' if balanced else 'random'} test set...")
    
    # Create generator for color pairs
    if balanced:
        # Generate all permutations (Red on Blue AND Blue on Red)
        color_pairs = list(itertools.permutations(COLOR_KEYS, 2))
        
        if samples_total < len(color_pairs):
            print(f"Warning: Total samples ({samples_total}) < unique color pairs ({len(color_pairs)}). Clipping.")
            selected_pairs = random.sample(color_pairs, samples_total)
        else:
            cycles = samples_total // len(color_pairs)
            remainder = samples_total % len(color_pairs)
            selected_pairs = color_pairs * cycles + random.sample(color_pairs, remainder)
            
        random.shuffle(selected_pairs)
    else:
        selected_pairs = []
        for _ in range(samples_total):
            bg = random.choice(COLOR_KEYS)
            fg = random.choice(COLOR_KEYS)
            while bg == fg:
                fg = random.choice(COLOR_KEYS)
            selected_pairs.append((bg, fg))

    for i, (bg_name, fg_name) in enumerate(selected_pairs):
        label = random.choice(ALL_SHAPES)
        
        image_id = f"test_{i:05d}"
        filename = f"{image_id}.png"
        
        # 1. Render
        img = render_sample(label, bg_name, fg_name, image_size)
        img.save(images_dir / filename)
        
        # 2. Generate Prompt and Metadata
        qa_data = generate_qa_pair(label, bg_name, fg_name)
        
        # 3. LLaVA Format (Input)
        llava_entry = {
            "id": image_id,
            "image": str(Path("images") / filename),
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{qa_data['question']}"
                },
                {
                    "from": "gpt",
                    "value": qa_data['full_answer']
                }
            ]
        }
        dataset_llava.append(llava_entry)
        
        # 4. Metadata Format (Ground Truth)
        meta_entry = {
            "id": image_id,
            "image_path": str(images_dir / filename),
            "config": {
                "bg_color": bg_name,
                "fg_color": fg_name,
                "object_content": label,
            },
            "task_type": qa_data['mode'],
            "ground_truth": qa_data['short_answer']
        }
        metadata_list.append(meta_entry)

    # Save
    with open(out_dir / "test_dataset.json", "w") as f:
        json.dump(dataset_llava, f, indent=2)

    with open(out_dir / "test_metadata.jsonl", "w") as f:
        for entry in metadata_list:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Done. Generated {len(dataset_llava)} samples in '{out_dir}'.")
    print(f"Ground truth metadata saved to '{out_dir}/test_metadata.jsonl'")

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Dual-Color Test Set Generator")
    parser.add_argument("--out", type=Path, default=Path("benchmark_dual_color"), help="Output directory")
    parser.add_argument("--count", type=int, default=500, help="Number of test samples") 
    parser.add_argument("--size", type=int, nargs=2, default=DEFAULT_SIZE, help="Image size (W H)")
    parser.add_argument("--balanced", action="store_true", default=True, help="Force balanced color pairs")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()
    
    generate_test_set(
        args.out, 
        args.count, 
        balanced=args.balanced, 
        image_size=tuple(args.size), 
        seed=args.seed
    )

if __name__ == "__main__":
    main()