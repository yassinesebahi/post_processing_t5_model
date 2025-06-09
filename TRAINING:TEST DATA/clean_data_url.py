import json
import re

input_path = "t5_training_data_large.jsonl"
output_path = "t5_training_data_no_urls.jsonl"

url_pattern = r"^(?:\.\./[^ ]*|ECLI:[^ ]+|vgl\.|https?://\S+)\s*"

cleaned_data = []

with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        item = json.loads(line)
        target = item.get("target", "").strip()

        # Clean prefix
        cleaned_target = re.sub(url_pattern, '', target)
        item["target"] = cleaned_target
        cleaned_data.append(item)

with open(output_path, "w", encoding="utf-8") as outfile:
    for ex in cleaned_data:
        outfile.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Cleaned {len(cleaned_data)} examples → saved to {output_path}")
