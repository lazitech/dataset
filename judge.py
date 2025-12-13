import requests
import json

def get_judge_prompt_v2(query, ground_truth, prediction):
    return f"""
You are an intelligent evaluator for a Visual Question Answering (VQA) task.
Your goal is to determine if the Model Prediction matches the Ground Truth.

Input:
- Question: {query}
- Ground Truth: {ground_truth}
- Model Prediction: {prediction}

Evaluation Criteria:
1. **Core Information Focus (CRITICAL)**:
   - Your primary job is to check if the *answer to the question* is correct.
   - **Ignore Extra Details**: If the Ground Truth is simple (e.g., "Purple") and the Prediction is detailed (e.g., "Purple with small darker dots"), this is **CORRECT**. The model sees the actual image; the Ground Truth might be simplified.
   - **No Penalty for Specificity**: Do NOT penalize the model for describing shapes (e.g., "letter X") or textures (e.g., "noise", "dots") that are not in the Ground Truth, provided the core attribute (color/object) is correct.

2. **Color Flexibility**:
   - Treat visually similar colors as matches.
   - **Cyan / Teal / Turquoise** -> MATCH.
   - **Purple / Violet / Lavender** -> MATCH.
   - **Green / Olive / Lime** -> MATCH.

3. **Numerical Accuracy (Conditional)**:
   - **Integer GT**: Exact match required (e.g., GT="2020", Pred="2019" -> WRONG).
   - **Decimal GT**: Allow 5% relative error.

4. **Unit & Format**: 
   - Ignore missing/extra units or symbols (e.g., "%", "$").

Output Format:
Return a valid JSON object ONLY:
{{
    "correct": boolean,
    "reason": "Brief explanation focused on semantic matching."
}}
"""

def judge_answer(query, gt, pred, api_key):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    payload = {
        "model": "Qwen/Qwen2.5-14B-Instruct", 
        "messages": [
            {
                "role": "user",
                "content": get_judge_prompt_v2(query, gt, pred)
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": { "type": "json_object" } 
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        else:
            return {"correct": False, "reason": f"API Error: {data}"}
    except Exception as e:
        return {"correct": False, "reason": str(e)}

# 使用示例
if __name__ == "__main__":

    q = "Identify the background color behind the brown character 'B'."
    gt = "The background is cyan."
    pred = "The background color behind the brown character 'B' is teal."
    
    MY_API_KEY = "sk-hmkocpctxwanyoczcaqktrmrfnbrzzbrjufgvvfatutlsdre" 
    
    result = judge_answer(q, gt, pred, MY_API_KEY)
    print(f"Test Result: {result}")
