import requests

def getprompt(query, ground_truth, prediction):
    return f"""
You are an intelligent evaluator for a Visual Question Answering (VQA) task.
Your goal is to determine if the Model Prediction matches the Ground Truth.

Input:
- Question: {query}
- Ground Truth: {ground_truth}
- Model Prediction: {prediction}

Evaluation Criteria:
1. **Numerical Accuracy (Conditional Rule)**:
   - **Case A: Integer Ground Truth** (e.g., "42", "2020", "100"): 
     The numeric value in the prediction MUST match the Ground Truth **EXACTLY**. No deviation is allowed.
     (e.g., GT="2020", Pred="2019" -> WRONG; GT="50", Pred="50" -> CORRECT).
   - **Case B: Decimal Ground Truth** (e.g., "42.5", "0.123", "1.5"): 
     Allow a relative error of up to **5%**.
     (e.g., GT="12.5", Pred="12.4" -> CORRECT).

2. **Unit & Format Flexibility**: 
   - Ignore missing or extra units/symbols if the number adheres to Rule 1.
   - Example: GT="50" vs Pred="50%" -> CORRECT.
   - Example: GT="$100" vs Pred="100 dollars" -> CORRECT.

3. **Verbose Answers**: 
   - Extract the core answer from full sentences.
   - Example: GT="Blue" vs Pred="The bar color is Blue" -> CORRECT.

4. **Synonyms**: 
   - Accept common synonyms (e.g., GT="Rose" vs Pred="Increased" -> CORRECT).

Output Format:
Return a valid JSON object(Just the JSON not ```json and not anything else) with the following structure:
{{
    "correct": boolean,
    "reason": "Brief explanation (e.g., 'Integer match exact' or 'Decimal within 5% range')"
}}
"""


url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": getprompt('Which retailers have a Highest sum of social media presence in the United Kingdom?','[\'Facebook users\']','Online shoppers')
        }
    ],
    "stream": False,
    "max_tokens": 4096,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "stop": [],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": { "type": "text" },
    "tool_choice": "none"
}
headers = {
    "Authorization": "Bearer sk-hmkocpctxwanyoczcaqktrmrfnbrzzbrjufgvvfatutlsdre",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

data = response.json()
answer = data["choices"][0]["message"]["content"]
print(answer)


