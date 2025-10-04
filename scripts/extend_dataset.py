import asyncio
from googletrans import Translator
import csv

async def main():
    with open("generated_extremist_unique.csv", mode="r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    new_rows = []
    print(f"Translating {len(rows)} rows...")
    for row in rows:
        text = row[0]
        translated_text = await translate_text(text)
        if translated_text:
            new_rows.append([translated_text, 1])

    print(f"Translated {len(new_rows)} rows.")
    with open("generated_extremist_unique.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)


async def translate_text(text):
    print("Translating:", text)
    translator = Translator()
    try:
        translation = await translator.translate(text, dest='fr')
        translation = await translator.translate(translation.text, dest='en')
        return translation.text
    except Exception as e:
        print("Translation error:", e)
        return None

if __name__ == "__main__":
    asyncio.run(main())
