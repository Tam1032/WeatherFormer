# toolformer_gradio_app.py
import os
import re
import time
import random
import requests
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

from utils import get_weather
# ==============================
# TEXT POST-PROCESSING
# ==============================

def remove_tool_calls(text: str) -> str:
    """Remove all [Get_weather(...) -> ...] patterns from the text."""
    # Remove the entire tool call: [Get_weather(...) -> ...]
    clean = re.sub(r'\[Get_weather\([^)]+\)\s*->\s*[^]]+\]', '', text)
    # Clean up extra spaces
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

# ==============================
# MODEL & GENERATION (same as before)
# ==============================

def load_model():
    adapter_path = "./lora-toolformer-qwen2-7b"
    if not os.path.exists(adapter_path):
        raise RuntimeError(f"LoRA adapter not found at {adapter_path}")

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
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def generate_with_real_tools(question: str, max_new_tokens: int = 256) -> str:
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

# ==============================
# GRADIO INTERFACE (UPDATED)
# ==============================

with gr.Blocks(title="ToolFormer-Qwen2: Dual Output") as demo:
    gr.Markdown("## 🌤️ ToolFormer-Qwen2 with Real Weather API")
    gr.Markdown("Get **two answers**: one with tool call (for debugging), one clean (for end users).")

    question = gr.Textbox(label="Your Question", placeholder="e.g., Is it raining in London?")
    submit = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column():
            answer_clean = gr.Textbox(label="Answer (Tool Call Removed)", interactive=False, lines=5)
        with gr.Column():
            answer_with_tool = gr.Textbox(label="Answer with Tool Call", interactive=False, lines=5)

    examples = [
        ["Will it snow in Tokyo today?"],
        ["Is the weather good for hiking in Zurich?"],
        ["Should I bring a jacket to Seoul?"]
    ]
    gr.Examples(examples, inputs=question)

    def respond(q):
        if not q.strip():
            return "", ""
        full_answer = generate_with_real_tools(q)
        clean_answer = remove_tool_calls(full_answer)
        return clean_answer, full_answer

    submit.click(respond, inputs=question, outputs=[answer_clean, answer_with_tool])

# ==============================
# LAUNCH
# ==============================

if __name__ == "__main__":
    # Optional: disable TF32 for reproducibility (not required)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("Launching Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860)