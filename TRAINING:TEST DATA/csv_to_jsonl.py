import pandas as pd
import json

csv_path = "normalization_training_set.csv"
df = pd.read_csv(csv_path, delimiter=';')
jsonl_path = "ground_truth_from_supervisor.jsonl"

with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        input_text = row["raw_citation"].strip()
        target_text = row["normalized_citation"].strip()
        json_line = {"input": input_text, "target": target_text}
        f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

print(f"Saved {len(df)} examples to: {jsonl_path}")
