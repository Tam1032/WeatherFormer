import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_ID = os.environ["ADAPTER_ID"]
OUT_DIR = "models/weatherformer-merged"

# Load the base in 16-bit, NOT 4-bit: merging into a quantized base loses precision.
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_ID).merge_and_unload()
model.save_pretrained(OUT_DIR, safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen2-7B").save_pretrained(OUT_DIR)
print(f"Merged model saved to {OUT_DIR}")
