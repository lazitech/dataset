"""
Author : 吕安哲
Description:
    通用数据集生成脚本。负责从 HuggingFaceM4/the_cauldron 数据集中提取文本描述图像数据，
    并生成包含图像和对应描述的 JSON 文件以及图像文件夹 images/。
"""
import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import io

NUM_SAMPLES = 2000 
OUTPUT_DIR = "./data/general_data"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_PATH = os.path.join(OUTPUT_DIR, "general_train.json")

os.makedirs(IMG_DIR, exist_ok=True)

PROMPTS = [
    "Describe this image.",
    "What is shown in this picture?",
    "Provide a caption for this image.",
    "Describe the scene and any text visible.",
    "What does this image display?",
    "Write a short description of the image."
]

def main():

    try:
        dataset = load_dataset(
            "HuggingFaceM4/the_cauldron", 
            "textcaps", 
            split="train", 
            streaming=True
        )
    except Exception as e:
        print(f"连接失败: {e}")
        return

    buffer = []
    try:
        for i, item in tqdm(enumerate(dataset), total=3000):
            if i >= 3000: break
            if 'images' in item and len(item['images']) > 0 and 'texts' in item and len(item['texts']) > 0:
                buffer.append(item)
    except Exception as e:
        print(f"读取数据流错误: {e}")
    
    if len(buffer) < NUM_SAMPLES:
        selected_items = buffer
    else:
        selected_items = random.sample(buffer, NUM_SAMPLES)

    final_json = []

    for idx, item in tqdm(enumerate(selected_items), total=len(selected_items)):
        try:
            img_obj = item['images'][0] 
            img_filename = f"general_textcaps_{idx:05d}.jpg"
            img_path = os.path.join(IMG_DIR, img_filename)
            if img_obj.mode != "RGB":
                img_obj = img_obj.convert("RGB")
            img_obj.save(img_path)
            caption = item['texts'][0] 
            human_query = "<image>\n" + random.choice(PROMPTS)
            conversations = [
                {
                    "from": "human",
                    "value": human_query
                },
                {
                    "from": "gpt",
                    "value": caption
                }
            ]
            entry = {
                "id": f"general_textcaps_{idx:05d}",
                "image": f"images/{img_filename}", 
                "conversations": conversations
            }
            final_json.append(entry)
            

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
