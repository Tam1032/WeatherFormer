# finetune_qwen2_peft.py
import os
import re
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# ----------------------------
# 1. Load your filtered data (Q:/A: format)
# ----------------------------

def load_toolformer_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r'\n(?=Q: )', text.strip())
    prompts = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 2 and lines[0].startswith("Q:") and lines[1].startswith("A:"):
            q = lines[0][2:].strip()
            a = lines[1][2:].strip()
            # Qwen doesn't require special prompt format, but we can use simple Q/A
            # Alternatively, use chat template (commented below)
            prompt = f"Q: {q}\nA: {a}"
            prompts.append({"text": prompt})
    
    return Dataset.from_list(prompts)

# ----------------------------
# 2. Model & Tokenizer (Qwen2-7B)
# ----------------------------

def create_model_and_tokenizer():
    model_name = "Qwen/Qwen2-7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        trust_remote_code=True,  # Required for Qwen
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ✅ Qwen-specific LoRA target modules
    # Based on Qwen2 architecture (similar to Llama, but verify)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # Works for Qwen2
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

# ----------------------------
# 3. Training
# ----------------------------

def main():
    data_path = "filtered_data.txt"          # Your filtered file
    output_dir = "./lora-toolformer-qwen2-7b"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    dataset = load_toolformer_data(data_path)
    print(f"Loaded {len(dataset)} examples")

    # Model & tokenizer
    model, tokenizer = create_model_and_tokenizer()

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting Qwen2-7B PEFT finetuning...")
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Qwen2-7B LoRA adapter saved to: {output_dir}")

if __name__ == "__main__":
    main()