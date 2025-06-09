
import json
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
from tqdm import tqdm
import sys
from common import tokenizer, id2tagNER

def load_model(model_path='model/bert_multiclass'):
    model = BertForTokenClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return model

def batch_tag(model, sentences, max_length=128):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    word_ids_batch = [inputs.word_ids(batch_index=i) for i in range(len(sentences))]
    tokens_batch = [tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(len(sentences))]

    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)

    results = []
    for i, sentence in enumerate(sentences):
        tokens = tokens_batch[i]
        word_ids = word_ids_batch[i]
        preds = predictions[i]

        tagged = []
        previous_word_idx = None
        for token, pred, word_idx in zip(tokens, preds, word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            tag = id2tagNER[pred.item()]
            word = tokenizer.convert_tokens_to_string([token]).strip()
            tagged.append({"token": word, "tag": tag})
            previous_word_idx = word_idx
        results.append(tagged)
    return results

def tag_file(input_txt_path, output_json_path, batch_size=64):
    with open(input_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    model = load_model()
    output = []

    for i in tqdm(range(0, len(lines), batch_size), desc="Batch tagging", dynamic_ncols=True, mininterval=0.5):
        batch = lines[i:i + batch_size]
        tagged_batch = batch_tag(model, batch)
        for j, tagged in enumerate(tagged_batch):
            output.append({
                "line_id": f"L{i + j + 1:04}",
                "original_text": batch[j],
                "tagged_tokens": tagged
            })

    with open(output_json_path, "w", encoding="utf-8") as out:
        json.dump(output, out, indent=2, ensure_ascii=False)

    print(f"Batch tagging complete. Output saved to {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_tagger.py <input_file.txt> <output_file.json>")
        sys.exit(1)

    tag_file(sys.argv[1], sys.argv[2])
