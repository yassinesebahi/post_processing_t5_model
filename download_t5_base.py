from transformers import T5Tokenizer, T5ForConditionalGeneration

def download_and_save_t5_base():
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    save_path = "t5-base-model"
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"T5-Base model and tokenizer saved to: {save_path}")

if __name__ == "__main__":
    download_and_save_t5_base()
