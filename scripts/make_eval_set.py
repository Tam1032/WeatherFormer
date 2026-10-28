"""Build eval/questions.jsonl from cities the model never saw in training."""
import csv
import json
import random

# None of these appear in data/cities.csv (the script checks).
HELD_OUT_CITIES = [
    ("Hue", "Vietnam"), ("Can Tho", "Vietnam"), ("Nha Trang", "Vietnam"), ("Jeonju", "South Korea"),
    ("Ulsan", "South Korea"), ("Hiroshima", "Japan"), ("Kanazawa", "Japan"), ("Sendai", "Japan"),
    ("Kraków", "Poland"), ("Gdańsk", "Poland"), ("Bilbao", "Spain"), ("Granada", "Spain"),
    ("Bordeaux", "France"), ("Strasbourg", "France"), ("Leipzig", "Germany"), ("Bologna", "Italy"),
    ("Tromsø", "Norway"), ("Uppsala", "Sweden"), ("Hobart", "Australia"), ("Halifax", "Canada"),
    ("Boise", "United States"), ("Anchorage", "United States"), ("Dunedin", "New Zealand"),
    ("Marrakesh", "Morocco"), ("Mendoza", "Argentina"), ("Medellín", "Colombia"), ("Arequipa", "Peru"),
    ("Chiang Rai", "Thailand"), ("Coimbra", "Portugal"), ("Kunming", "China"), ("Tainan", "Taiwan"),
]

NO_TOOL_QUESTIONS = [
    "What is the capital of Japan?",
    "How many days are there in a leap year?",
    "Who wrote Romeo and Juliet?",
    "What is 15 multiplied by 12?",
    "What is the boiling point of water at sea level in Celsius?",
    "How many continents are there?",
    "What language is spoken in Brazil?",
    "What is the largest planet in the solar system?",
    "Translate 'thank you' into French.",
    "What does CPU stand for?",
    "How many minutes are in three hours?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "Name a primary color.",
    "What is the square root of 81?",
    "Which ocean is the largest?",
    "What year did World War II end?",
    "How many legs does a spider have?",
    "What is the currency of the United Kingdom?",
    "What is photosynthesis in one sentence?",
    "Is a tomato a fruit or a vegetable?",
    "What is the tallest mountain in the world?",
    "How do you say hello in Spanish?",
    "What gas do humans exhale?",
    "How many sides does a hexagon have?",
]

N_TOOL_QUESTIONS = 150


def main():
    with open("data/cities.csv", encoding="utf-8") as f:
        train_cities = {(r["city"], r["country"]) for r in csv.DictReader(f)}
    overlap = [c for c in HELD_OUT_CITIES if c in train_cities]
    assert not overlap, f"cities seen in training: {overlap}"

    with open("data/weather_templates.json", encoding="utf-8") as f:
        templates = json.load(f)

    random.seed(42)
    rows = []
    for _ in range(N_TOOL_QUESTIONS):
        city, country = random.choice(HELD_OUT_CITIES)
        question = random.choice(templates).replace("{city}", city).replace("{country}", country)
        rows.append({"question": question, "expect_call": True, "city": city, "country": country})
    rows += [{"question": q, "expect_call": False, "city": None, "country": None} for q in NO_TOOL_QUESTIONS]

    with open("eval/questions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions to eval/questions.jsonl")


if __name__ == "__main__":
    main()
