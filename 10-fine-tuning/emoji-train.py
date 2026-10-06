import modal

volume = modal.Volume.from_name("unsloth-emoji-storage", create_if_missing=True)

unsloth_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth", "transformers", "trl", "peft", 
        "accelerate", "bitsandbytes", "datasets"
    )
)

app = modal.App("unsloth-emoji-finetune")

@app.function(
    image=unsloth_image,
    gpu="A100", 
    volumes={"/data": volume},
    timeout=3600,
)
def train_emoji_model():
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # --- CONFIG ---
    max_seq_length = 2048
    model_name = "unsloth/llama-3-8b-bnb-4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # --- EMOJI DATASET PREP (UPDATED) ---
    prompt_style = """Below is a sentence. Translate the content and sentiment into emojis.

### Sentence:
{}

### Emojis:
{}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        inputs  = examples["text"]
        outputs = examples["emojification"] # Using the 'emojification' column
        texts = []
        for input_text, output_emoji in zip(inputs, outputs):
            text = prompt_style.format(input_text, output_emoji) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    # Using the verified text-to-emoji dataset
    dataset = load_dataset("sjoerdbodbijl/text-to-emoji", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # --- TRAINING ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 80, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "/data/outputs",
        ),
    )

    trainer.train()

    # --- SAVE ---
    model.save_pretrained("/data/emoji_model")
    tokenizer.save_pretrained("/data/emoji_model")
    volume.commit()
    print("✨ Emoji training complete with new dataset!")

@app.local_entrypoint()
def main():
    train_emoji_model.remote()