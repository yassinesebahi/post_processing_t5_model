import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from difflib import SequenceMatcher
import pandas as pd


model_path = "t5-normalization-output/final_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
file_path = "synthetic_test_set.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []
for item in tqdm(data, desc="Evaluating model"):
    input_text = item["input"]
    expected_output = item["target"]

    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=128)
    predicted_output = tokenizer.decode(output[0], skip_special_tokens=True)

    match = predicted_output.strip() == expected_output.strip()
    similarity = SequenceMatcher(None, predicted_output.strip(), expected_output.strip()).ratio()

    results.append({
        "input": input_text,
        "expected_output": expected_output,
        "predicted_output": predicted_output.strip(),
        "match": match,
        "similarity": round(similarity, 3)
    })

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("Evaluation results saved to evaluation_results.csv")
