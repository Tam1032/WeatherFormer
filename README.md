# 🌦️ WeatherFormer: Teaching LLMs to Use Real Weather APIs

**WeatherFormer** extends the [ToolFormer](https://arxiv.org/abs/2302.04761) framework to teach large language models (LLMs) how to **reliably use real-world weather APIs** — instead of hallucinating time-sensitive forecasts.

Using **Qwen2-7B** and **real Open-Meteo API data**, this project demonstrates end-to-end:
- Synthetic data generation with tool calls
- Self-supervised filtering based on model confidence
- Parameter-efficient fine-tuning (PEFT/LoRA)
- Real-time inference with **on-the-fly API execution**

---

## ✨ Features

- **Accurate weather responses**: No more guessing — the model calls a real API when needed.
- **Dual output modes**:
  - **With tool trace**: `[Get_weather(City, Country) → 28°C, clear] Yes, it's a great day...`
  - **Clean answer**: `Yes, it's a great day...`
- **Memory-efficient**: 4-bit quantization + LoRA fine-tuning (runs on a single 24GB GPU).
- **Easy deployment**: Gradio web UI included.

---

## 🛠️ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Running inference
```bash
python app.py
```
