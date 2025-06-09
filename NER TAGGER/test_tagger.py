# test_tagger.py
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')

id2tagNER = ['O', 'B-Autor', 'I-Autor', 'B-Aktenzeichen', 'I-Aktenzeichen', 'B-Auflage', 'I-Auflage',
             'B-Datum', 'I-Datum', 'B-Editor', 'B-Gesetz', 'I-Gesetz', 'B-Gericht', 'I-Gericht',
             'B-Jahr', 'B-Nummer', 'I-Nummer', 'B-Randnummer', 'I-Randnummer', 'B-Paragraph',
             'I-Paragraph', 'B-Seite-Beginn', 'I-Seite-Beginn', 'B-Seite-Fundstelle', 'B-Titel',
             'I-Titel', 'B-Zeitschrift', 'I-Zeitschrift', 'I-Editor', 'I-Seite-Fundstelle',
             'B-Wort:Auflage', 'I-Wort:Auflage', 'B-Wort:aaO', 'I-Wort:aaO']

def tag_sentence(text):
    model = BertForTokenClassification.from_pretrained('model/bert_multiclass', local_files_only=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    tokens = list(map(tokenizer.decode, inputs['input_ids'][0]))
    logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_tags = [id2tagNER[i.item()] for i in predictions[0]]

    for token, tag in zip(tokens, predicted_tags):
        print(f"{token} -> {tag}")

if __name__ == "__main__":
    tag_sentence("einschließlich der anzurechnenden Untersuchungshaft, vgl. BGH NStZ-RR 2008, 182")
