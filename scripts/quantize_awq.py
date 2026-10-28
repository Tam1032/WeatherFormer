import random

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC_DIR = "models/weatherformer-merged"
OUT_DIR = "models/weatherformer-awq"
NUM_SAMPLES = 128
MAX_LEN = 512

# Calibrate on the same Q/A format the model was fine-tuned on.
with open("data/filtered_data.txt", encoding="utf-8") as f:
    blocks = [b.strip() for b in f.read().strip().split("\n\n") if b.strip()]
random.seed(0)
samples = random.sample(blocks, NUM_SAMPLES)

model = AutoModelForCausalLM.from_pretrained(SRC_DIR, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(SRC_DIR)
ds = Dataset.from_dict({"text": samples}).map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LEN, add_special_tokens=False),
    remove_columns=["text"],
)

recipe = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_LEN, num_calibration_samples=NUM_SAMPLES)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
print(f"AWQ model saved to {OUT_DIR}")
