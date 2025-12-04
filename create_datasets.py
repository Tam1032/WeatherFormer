import requests
import time
import random
import csv
import json
import ollama
from typing import List, Dict

from utils import get_weather

# ==============================
# CONFIG
# ==============================

CITIES_CSV_PATH = "cities.csv"
TEMPLATE_FILE = "weather_templates.json"
MODEL_NAME = "gpt-oss:20b"

# ==============================
# TEMPLATE GENERATION
# ==============================

def generate_templates_with_llm(num_templates: int = 200) -> List[str]:
    prompt = f"""
    You are generating training data for a ToolFormer-style dataset.

    Generate exactly {num_templates} UNIQUE user questions that *require* checking the current weather.

    STRICT RULES:
    - Must contain placeholders: {{city}}, {{country}}
    - Must require weather info to answer (temperature, rain, heat, wind, snow, outdoor suitability, clothing, safety)
    - No generic travel questions
    - Avoid mentioning humidity or air quality
    - Avoid specific dates or times, especially past or future dates (e.g. "tomorrow", "next week")
    - Sound natural and conversational

    Return only the questions, one per line.
    """
    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    lines = response["response"].split("\n")

    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            line = line.split(".", 1)[-1].strip()
        if "{city}" in line and "{country}" in line:
            cleaned.append(line)
    print(f"Generated {len(cleaned)} templates.")
    return cleaned

def save_templates(templates: List[str]):
    with open(TEMPLATE_FILE, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2)

def load_templates() -> List[str]:
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ==============================
# CITY LOADING
# ==============================

def load_cities() -> List[Dict[str, str]]:
    cities = []
    with open(CITIES_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append(row)
    return cities

# ==============================
# TOOLFORMER STYLE ANSWER GENERATION
# ==============================

def generate_toolformer_style_answer(question: str, weather: dict) -> str:
    """
    Generates ToolFormer-style answer:
    <neutral lead-in>, [tool_call -> tool_output] <natural continuation>
    """
    city = weather["city"]
    country = weather["country"]
    temp = weather["temperature"]
    cond = weather["condition"]
    wind = weather["wind_speed"]

    tool_call = f"Get_weather({city}, {country})"
    tool_output = f"{temp}°C, {cond}, wind {wind} km/h"

    neutral_intro = random.choice(NEUTRAL_INTROS)

    # Prompt the LLM to generate natural continuation
    prompt = f"""
    You are generating a ToolFormer-style answer for a weather question.

    User question:
    \"\"\"{question}\"\"\"

    Weather tool output:
    \"\"\"[{tool_call} -> {tool_output}]\"\"\"

    Write a short, neutral continuation immediately after the tool call.

    STRICT RULES:
    - Do NOT assume anything before the tool call.
    - After the tool call, give only the information that directly answers the question.
    - Use temperature, condition, or wind only if they are relevant to the user’s question.
    - Keep the continuation concise, natural, and free of unnecessary details.
    - Do NOT restate the full tool output unless needed.
    - Do NOT explain; just answer in a compact, fluent way.
    - Return only the continuation text, nothing else.
    """
    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    continuation = response["response"].strip()

    return f"[{tool_call} -> {tool_output}] {continuation}"

# ==============================
# MAIN DATASET GENERATOR
# ==============================

def generate_dataset(num_samples: int = 2000):
    cities = load_cities()
    templates = load_templates()
    dataset = []

    for i in range(num_samples):
        c = random.choice(cities)
        city = c["city"]
        country = c["country"]

        template = random.choice(templates)
        question = template.replace("{city}", city).replace("{country}", country)

        weather = get_weather(city, country)
        if not weather:
            continue

        toolformer_answer = generate_toolformer_style_answer(question, weather)
        entry = f"Q: {question}\nA: {toolformer_answer}"

        dataset.append(entry)

        time.sleep(0.08 + random.random() * 0.05)
        if i % 50 == 0:
            print(f"{i}/{num_samples} generated...")

    return dataset

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    templates = generate_templates_with_llm(num_templates=200)
    save_templates(templates)

    dataset = generate_dataset(num_samples=5000)

    with open("weather_toolformer_dataset.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(dataset))

    print("Dataset generation complete.")
