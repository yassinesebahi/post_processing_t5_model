
import json

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_valid(fields):
    return sum(1 for v in fields.values() if v and str(v).strip()) >= 2

def build_input(fields):
    parts = []
    for key in ['court', 'year', 'page_start', 'case_number', 'paragraph', 'law']:
        value = fields.get(key)
        if value and str(value).strip():
            parts.append(f"{key}={value.strip()}")
    return "normalize: " + "; ".join(parts)

def build_target(entry):
    return entry.get("original_text", "").strip()

def generate_t5_data(input_json, output_jsonl):
    data = load_data(input_json)
    output = []

    for entry in data:
        fields = entry.get("structured_fields", {})
        if is_valid(fields):
            input_str = build_input(fields)
            target_str = build_target(entry)
            output.append({"input": input_str, "target": target_str})

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(output)} training pairs at: {output_jsonl}")

if __name__ == "__main__":
    generate_t5_data("structured_output.json", "t5_training_data_large.jsonl")
