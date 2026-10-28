import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8080")


def respond(question: str):
    if not question.strip():
        return "", "", ""
    try:
        r = requests.post(f"{API_URL}/v1/ask", json={"question": question}, timeout=120)
        r.raise_for_status()
    except requests.RequestException as exc:
        return f"Error: {exc}", "", ""
    body = r.json()
    return body["answer"], body["answer_with_trace"], f"{body['latency_ms']:.0f} ms"


with gr.Blocks(title="WeatherFormer") as demo:
    gr.Markdown("## 🌤️ WeatherFormer: Qwen2-7B + LoRA with a real weather API")
    question = gr.Textbox(label="Your Question", placeholder="e.g., Is it raining in London, United Kingdom?")
    submit = gr.Button("Ask", variant="primary")
    with gr.Row():
        answer_clean = gr.Textbox(label="Answer (Tool Call Removed)", interactive=False, lines=5)
        answer_with_tool = gr.Textbox(label="Answer with Tool Call", interactive=False, lines=5)
    latency = gr.Textbox(label="Latency", interactive=False)
    gr.Examples([["Will it snow in Tokyo, Japan today?"],
                 ["Is the weather good for hiking in Zurich, Switzerland?"],
                 ["Should I bring a jacket to Seoul, South Korea?"]], inputs=question)
    submit.click(respond, inputs=question, outputs=[answer_clean, answer_with_tool, latency])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
