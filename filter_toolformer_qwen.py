import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

# ----------------------------
# ToolFormer Filtering (Corrected)
# ----------------------------

import torch
from torch.nn.utils.rnn import pad_sequence

def toolformer_filter(
    model,
    tokenizer,
    originals: list[str],
    no_responses: list[str],
    with_responses: list[str],
    api_end_str: str = "]",
    threshold: float = 1.0,
    device: str = "cuda",
    batch_size: int = 8     # <<< change this depending on your GPU
):
    model.eval()
    pad_id = tokenizer.pad_token_id
    api_end_id = tokenizer.encode(api_end_str, add_special_tokens=False)[0]

    def tokenize(texts):
        return [torch.tensor(tokenizer.encode(t, add_special_tokens=False)) for t in texts]

    orig_tok = tokenize(originals)
    no_resp_tok = tokenize(no_responses)
    with_resp_tok = tokenize(with_responses)

    def get_loss(tokens, logits):
        logits = logits[:, :-1]
        targets = tokens[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        true_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return -true_log_probs

    def get_weights(tokens):
        mask = (tokens == api_end_id)
        after_api = mask.cumsum(dim=1) > 0
        weights = torch.cat(
            [torch.zeros_like(after_api[:, :1]), after_api[:, :-1]],
            dim=1
        )
        steps_after = weights.cumsum(dim=1) * weights
        weights = torch.clamp(1.0 - 0.2 * steps_after, min=0.0)
        return weights[:, 1:]

    keeps = []

    N = len(originals)
    for i in range(0, N, batch_size):
        # slice batch
        orig_batch = orig_tok[i:i+batch_size]
        no_batch = no_resp_tok[i:i+batch_size]
        with_batch = with_resp_tok[i:i+batch_size]

        # pad inside CPU first
        orig_padded = pad_sequence(orig_batch, batch_first=True, padding_value=pad_id).to(device)
        no_padded = pad_sequence(no_batch, batch_first=True, padding_value=pad_id).to(device)
        with_padded = pad_sequence(with_batch, batch_first=True, padding_value=pad_id).to(device)

        with torch.no_grad():
            orig_logits = model(orig_padded).logits
            no_logits = model(no_padded).logits
            with_logits = model(with_padded).logits

        orig_loss = get_loss(orig_padded, orig_logits)
        no_loss = get_loss(no_padded, no_logits)
        with_loss = get_loss(with_padded, with_logits)

        orig_weights = get_weights(orig_padded)
        no_weights = get_weights(no_padded)
        with_weights = get_weights(with_padded)

        orig_w = (orig_weights * orig_loss).sum(dim=1)
        no_w = (no_weights * no_loss).sum(dim=1)
        with_w = (with_weights * with_loss).sum(dim=1)

        loss_minus = torch.minimum(orig_w, no_w)
        keep_batch = (loss_minus - with_w) >= threshold

        keeps.append(keep_batch.cpu())

        # free GPU memory for this batch
        del orig_padded, no_padded, with_padded
        del orig_logits, no_logits, with_logits
        torch.cuda.empty_cache()

    return torch.cat(keeps, dim=0)


# ----------------------------
# Parse Your Dataset
# ----------------------------

def parse_dataset(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into blocks
    blocks = re.split(r'\n(?=Q: )', content.strip())
    originals, no_responses, with_responses, valid_qas = [], [], [], []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 2 or not lines[0].startswith("Q:") or not lines[1].startswith("A:"):
            continue

        question = lines[0][2:].strip()
        full_answer = lines[1][2:].strip()  # e.g. "[api] natural answer"

        # Extract: [Get_weather(...) -> ...]
        api_with_response_match = re.search(r'(\[Get_weather\([^)]+\)\s*->\s*[^]]+\])', full_answer)
        if not api_with_response_match:
            continue

        api_with_response = api_with_response_match.group(1)  # [Get_weather(...) -> ...]
        # Extract just the API call: [Get_weather(...)]
        api_call_only = re.sub(r'\s*->\s*[^]]+', '', api_with_response)

        # Extract the natural language answer AFTER the ]
        after_api = full_answer[api_with_response_match.end():].strip()

        # ✅ CORRECT SEQUENCES:
        original = f"Q: {question}\nA: {after_api}"                     # Full human answer
        no_response = f"Q: {question}\nA: {api_call_only} {after_api}"               # API call only
        with_response = f"Q: {question}\nA: {api_with_response} {after_api}"  # API + response + answer

        originals.append(original)
        no_responses.append(no_response)
        with_responses.append(with_response)
        valid_qas.append((question, full_answer))

    return originals, no_responses, with_responses, valid_qas

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .txt file")
    parser.add_argument("--output", default="filtered.json")
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    # Load Qwen2-7B
    print("Loading Qwen2-7B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Parse data
    print(f"Parsing {args.input}...")
    originals, no_responses, with_responses, valid_qas = parse_dataset(args.input)
    print(originals[:2])  # Print first 5 examples for verification
    print(no_responses[:2])  # Print first 5 examples for verification
    print(with_responses[:2])  # Print first 5 examples for verification
    print(f"Found {len(originals)} examples")

    if not originals:
        print("No valid examples found!")
        exit(1)

    # Filter
    print("Running ToolFormer filtering...")
    keep_mask = toolformer_filter(
        model, tokenizer,
        originals, no_responses, with_responses,
        threshold=args.threshold,
        device="cuda"
    )

    # Save results
    # filtered = []
    # for i, keep in enumerate(keep_mask):
    #     if keep:
    #         filtered.append({
    #             "original": originals[i],
    #             "with_call": no_responses[i],
    #             "with_response": with_responses[i]
    #         })
    # with open(args.output, "w") as f:
    #     json.dump(filtered, f, indent=2, ensure_ascii=False)
    # Save filtered Q/A in original format
    filtered_qa = [valid_qas[i] for i, keep in enumerate(keep_mask) if keep]
    with open(args.output, "w", encoding="utf-8") as f:
        for q, a in filtered_qa:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")
    
    print(f"✅ Kept {len(filtered_qa)} / {len(originals)} examples")
    print(f"Saved to {args.output}")