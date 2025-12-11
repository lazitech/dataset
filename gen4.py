"""
Author: 吕安哲
Desciption:
    生成包含散点图的问答数据集（dataset4）。
    每个样本包含一张散点图，图中有若干个点，每个点有一个字母标签、一个数值和一个颜色。
    任务是根据图中点的数值对字母进行排序。
    训练集和测试集分别生成1000和200个样本。
    训练集的难度适中，测试集稍微难一点。
    生成的图像尺寸为754x868像素，点的颜色从预定义的颜色列表中随机选择。
    生成的样本以JSON格式保存，包含图像文件名、问题文本和答案。
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
from tqdm import tqdm
from typing import Dict, List, Any

BASE_DIR = "./dataset4"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train/images")
TRAIN_JSON_FILE = os.path.join(BASE_DIR, "train/train_dataset.json")

TEST_IMG_DIR = os.path.join(BASE_DIR, "test/images")
TEST_JSON_FILE = os.path.join(BASE_DIR, "test/test_dataset.json")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

COLOR_MAP: Dict[str, str] = {
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

def generate_one_sample(index, output_dir, mode="train", num_points=5, difficulty="medium"):

    raw_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    filtered_letters = [c for c in raw_letters if c not in ['I', 'O']]
    labels = random.sample(filtered_letters, num_points)
    
    all_colors = list(COLOR_MAP.keys())
    
    if num_points <= len(all_colors):
        chosen_color_names = random.sample(all_colors, num_points)
    else:
        chosen_color_names = random.choices(all_colors, k=num_points)
    chosen_hex_codes = [COLOR_MAP[name] for name in chosen_color_names]
    
    if difficulty == "easy":
        values = random.sample(range(10, 100), num_points)
    else:
        base = random.randint(30, 70)
        possible_values = list(range(base - 10, base + 10))
        values = random.sample(possible_values, num_points)

    data = list(zip(labels, values, chosen_color_names, chosen_hex_codes))
    
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    sorted_labels = [item[0] for item in sorted_data]
    
    fig = plt.figure(figsize=(754 / 100, 868 / 100), dpi=100)
    
    x_coords = range(len(labels))
    y_values = [d[1] for d in data]
    hex_colors = [d[3] for d in data] 
    
    ax = plt.gca()
    ax.vlines(x_coords, 0, y_values, colors=hex_colors, linestyles='solid', linewidth=1.8, alpha=0.6)
    plt.scatter(x_coords, y_values, c=hex_colors, s=270, alpha=0.95, edgecolors='none')
    
    for x, y, hex_c in zip(x_coords, y_values, hex_colors):
        plt.text(x, y + 4, f"{int(y)}", 
                 ha='center', va='bottom', 
                 fontsize=22, fontweight='bold', color=hex_c) 

    plt.xticks(x_coords, labels, fontsize=26, fontweight='bold')
    max_val = max(y_values)
    plt.ylim(0, max_val * 1.3) 
    
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    
    prefix = "train" if mode == "train" else "test"
    filename = f"{prefix}_chart_{index:05d}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=100)
    plt.close()

    entry = {
        "id": f"{prefix}_chart_{index:05d}",
        "image": filename,
    }

    question_text = "<image>\nAnalyze the scatter plot. Identify the value and color of each point associated with the letters, then sort the letters based on their values from highest to lowest."

    if mode == "train":
        explanation = "Let's analyze the chart. Looking at the numeric labels annotated above each data point:\n"
        for label, val, color_name, _ in data:
            explanation += f"- The point labeled {label} is {color_name} and represents the value {int(val)}.\n"
        
        explanation += f"Sorting these values from highest to lowest ({', '.join([str(d[1]) for d in sorted_data])}), the corresponding order of letters is: {', '.join(sorted_labels)}."
        
        entry["conversations"] = [
            {"from": "human", "value": question_text},
            {"from": "gpt", "value": explanation}
        ]

    else:
        entry["conversations"] = [
            {"from": "human", "value": question_text},
            {"from": "gpt", "value": f"The sorted order is: {', '.join(sorted_labels)}."} # 简短回答
        ]
    
        entry["ground_truth_list"] = sorted_labels 
        entry["ground_truth_values"] = [d[1] for d in sorted_data] # 可选：同时也给出数值答案

    return entry


def main():
    train_count =  1000  
    test_count = 200    
    
    train_dataset = []
    for i in tqdm(range(train_count)):
        diff = "hard" if random.random() < 0.4 else "easy"
        num_pts = random.randint(3, 8)
        
        entry = generate_one_sample(i, TRAIN_IMG_DIR, mode="train", num_points=num_pts, difficulty=diff)
        train_dataset.append(entry)

    with open(TRAIN_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)

    test_dataset = []
    for i in tqdm(range(test_count)):
        diff = "hard" if random.random() < 0.5 else "easy"
        num_pts = random.randint(3, 8)
        entry = generate_one_sample(i, TEST_IMG_DIR, mode="test", num_points=num_pts, difficulty=diff)
        test_dataset.append(entry)

    with open(TEST_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
