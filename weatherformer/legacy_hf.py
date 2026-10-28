"""Original inference loop from app.py, kept unchanged as the benchmark baseline."""
import os
import re

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from weatherformer.legacy_utils import get_weather


def load_model():
    adapter_path = os.getenv("ADAPTER_ID", "./lora-toolformer-qwen2-7b")  # local dir or HF Hub id

    print("Loading Qwen2-7B + LoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def generate_with_real_tools(question: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    prompt = f"Q: {question}\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()

    try:
        api_start_id = tokenizer.encode("[", add_special_tokens=False)[0]
        api_end_id = tokenizer.encode("]", add_special_tokens=False)[0]
    except:
        raise ValueError("Tokenizer must produce single tokens for '[' and ']'")

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

        if "\n" in tokenizer.decode(generated[0, input_ids.shape[1]:]):
            break  # same stop rule as the vLLM agent, so the benchmark is fair

        current_text = tokenizer.decode(generated[0], skip_special_tokens=False)

        match = re.search(r'\[Get_weather\(([^)]+)\)(?! ->)', current_text)
        if match and "->" not in match.group(0):
            args_str = match.group(1).strip()
            try:
                if "," not in args_str:
                    raise ValueError("Missing comma")
                city, country = [part.strip() for part in args_str.split(",", 1)]
                if not city or not country:
                    raise ValueError("Empty")
                weather_resp = get_weather(city, country)
                filled_call = f"[Get_weather({args_str}) -> {weather_resp}]"
                new_text = current_text.replace(match.group(0), filled_call)
                generated = tokenizer(new_text, return_tensors="pt").input_ids.to(model.device)
                continue
            except:
                pass

        if next_token.item() == tokenizer.eos_token_id:
            break

    full_output = tokenizer.decode(generated[0], skip_special_tokens=False)
    if "A:" in full_output:
        answer = full_output.split("A:", 1)[1].strip()
    else:
        answer = full_output.strip()
    answer = answer.replace("<|endoftext|>", "").strip()
    return answer
