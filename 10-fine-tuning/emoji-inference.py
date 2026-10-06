import modal
import os

# 1. Define the persistent storage and container environment
volume = modal.Volume.from_name("unsloth-emoji-storage", create_if_missing=True)

unsloth_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth", "transformers", "bitsandbytes", 
        "accelerate", "torch", "datasets", "trl"
    )
    .env({"UNSLOTH_DISABLE_FAST_GENERATION": "1"}) # Fixes the Shape Mismatch Error
)

app = modal.App("unsloth-emoji-comparison")

def get_model_response(model_name, instruction, is_custom_path=False):
    """Internal helper to load a model, generate, and then fully unload it."""
    from unsloth import FastLanguageModel
    import torch
    import gc

    print(f"--- Loading {model_name} ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    prompt_style = """Below is a sentence. Translate the content and sentiment into emojis.

### Sentence:
{}

### Emojis:
{}"""

    inputs = tokenizer(
        [prompt_style.format(instruction, "")], 
        return_tensors = "pt",
        padding = True
    ).to("cuda")

    outputs = model.generate(
        input_ids = inputs.input_ids,
        attention_mask = inputs.attention_mask,
        max_new_tokens = 50,
        use_cache = True
    )
    
    response = tokenizer.batch_decode(outputs)[0]
    
    # Cleanup logic to prevent RuntimeError shape mismatch on next load
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize() 
    
    try:
        return response.split("### Emojis:")[1].replace("<|endoftext|>", "").strip()
    except:
        return response.strip()

@app.function(image=unsloth_image, gpu="A100", volumes={"/data": volume})
def run_comparison(sentence: str):
    # 1. Get Base Model Result
    base_result = get_model_response("unsloth/llama-3-8b-bnb-4bit", sentence)
    
    # 2. Get Fine-Tuned Model Result (from Volume)
    ft_result = get_model_response("/data/emoji_model", sentence)
    
    return base_result, ft_result

@app.local_entrypoint()
def main(text: str = "I am so excited to go to the beach and eat ice cream!"):
    print(f"\nPROMPT: {text}\n" + "="*60)
    
    base, fine_tuned = run_comparison.remote(text)
    
    print(f"\n[ 🛠️  ORIGINAL BASE MODEL ]")
    print(f"Output: {base}")
    
    print("\n" + "-"*60)
    
    print(f"\n[ ✨ EMOJI FINE-TUNED MODEL ]")
    print(f"Output: {fine_tuned}")
    print("="*60 + "\n")