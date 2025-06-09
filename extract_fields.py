import json
import os

def extract_structured_fields(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        tagged_data = json.load(f)

    field_map = {
        "B-Gericht": "court",
        "B-Jahr": "year",
        "B-Zeitschrift": "journal",
        "B-Seite-Beginn": "page_start",
        "B-Aktenzeichen": "case_number",
        "I-Aktenzeichen": "case_number",
        "B-Gesetz": "law",
        "B-Paragraph": "paragraph",
        "I-Paragraph": "paragraph"
    }

    structured_entries = []
    for entry in tagged_data:
        field_values = {}
        for token_info in entry["tagged_tokens"]:
            tag = token_info["tag"]
            token = token_info["token"]
            field = field_map.get(tag)
            if not field:
                continue
            if field in field_values:
                field_values[field] += " " + token
            else:
                field_values[field] = token

        structured_entries.append({
            "line_id": entry["line_id"],
            "original_text": entry["original_text"],
            "structured_fields": field_values
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_entries, f, indent=2, ensure_ascii=False)

    print(f"Extracted fields saved to: {output_json_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_fields.py <input_tagged.json> <output_structured.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    extract_structured_fields(input_path, output_path)
