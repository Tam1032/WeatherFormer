from huggingface_hub import HfApi

REPO_ID = "tamH123/weatherformer-qwen2-7b-lora"  # HF username, may differ from GitHub

api = HfApi()
api.create_repo(REPO_ID, exist_ok=True)
api.upload_folder(
    folder_path="lora-toolformer-qwen2-7b",
    repo_id=REPO_ID,
    ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "scheduler*", "rng_state*"],
)
print(f"https://huggingface.co/{REPO_ID}")
