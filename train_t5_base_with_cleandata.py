import json
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Trainer
from transformers.training_args import TrainingArguments

def load_t5_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(lines)

def preprocess(example, tokenizer, max_input_length=128, max_target_length=128):
    model_input = tokenizer(example["input"], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        label = tokenizer(example["target"], max_length=max_target_length, truncation=True)
    model_input["labels"] = label["input_ids"]
    return model_input

def main():
    model_path = "t5-base-model"  
    data_path = "t5_training_data_no_urls.jsonl" 
    output_dir = "t5-normalization-output-v2"    
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    dataset = load_t5_data(data_path)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    tokenized_train = dataset["train"].map(lambda x: preprocess(x, tokenizer), batched=True)
    tokenized_eval = dataset["test"].map(lambda x: preprocess(x, tokenizer), batched=True)
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        logging_steps=1000,
        save_steps=99999,
        save_total_limit=1,
        prediction_loss_only=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
