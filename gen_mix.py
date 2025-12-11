"""
Author : 吕安哲
Description:
    数据预处理脚本。负责将生成的四组 Datasets (dataset1-4) 
    与 通用数据 (dataset0) 进行混合，生成最终的训练集文件 mixed_train_final.json
"""
import json
import os
import random
from tqdm import tqdm

DATASET_FOLDERS = [f"dataset{i}" for i in range(5)]
JSON_SUBPATH = os.path.join("train", "instruct.json")

OUTPUT_FILE = "mixed_train_final.json"

def main():
    all_data = []
    
    for folder_name in DATASET_FOLDERS:
        json_file_path = os.path.join(folder_name, JSON_SUBPATH)
        if not os.path.exists(json_file_path):
            print(f"找不到文件 {json_file_path}")
            continue
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for entry in data:
                if "image" in entry:
                    original_path = entry["image"]
                    parent_dir = os.path.dirname(json_file_path)
                    new_path = os.path.join(parent_dir, original_path)
                    entry["image"] = new_path.replace("\\", "/") 
                all_data.append(entry)

        except Exception as e:
            print(f"取 {json_file_path} 出错: {e}")

    print(f"合并完成，共计 {len(all_data)} 条数据。")

    random.seed(42)
    random.shuffle(all_data)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
