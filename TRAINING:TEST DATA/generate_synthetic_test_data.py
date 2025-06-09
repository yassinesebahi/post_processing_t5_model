import json
import random

courts = ["BGH", "LSG", "LAG", "FG", "SG", "VG", "OVG", "LSGH", "OLG"]
years = list(range(1990, 2024))
paragraphs = [
    "§ 242", "§ 41 Abs. 2", "§ 622 Abs. 5", "§ 134", "§ 9", "§ 613a Abs. 1 Satz 1"
]
laws = ["BGB", "StGB", "SGB III", "ZPO", "KSchG", "TzBfG", "GG"]
case_numbers = [
    "2 StR 123/21", "S 12 AL 45/18", "8 U 87/97", "11 AZR 101/20", "VII ZR 201/19"
]
page_starts = [str(random.randint(1, 999)).zfill(2) for _ in range(20)]

examples = []
for i in range(100):
    court = random.choice(courts)
    year = random.choice(years)
    paragraph = random.choice(paragraphs)
    law = random.choice(laws)
    case_number = random.choice(case_numbers)
    page_start = random.choice(page_starts)

    input_text = (
        f"court={court}; year={year}; page_start={page_start}; "
        f"case_number={case_number}; paragraph={paragraph}; law={law}"
    )

    normalized = f"{paragraph} {law}"

    examples.append({
        "input": input_text,
        "target": normalized
    })

with open("t5_test_data.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print("Synthetic test data written to t5_test_data.jsonl")
