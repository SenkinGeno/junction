from nltk.corpus import wordnet
import nltk
import csv
nltk.download('wordnet')

def get_synonym(word):
    syns = wordnet.synsets(word)
    if syns:
        lemmas = syns[0].lemmas()
        if lemmas:
            return lemmas[0].name().replace('_', ' ')
    return word


def main():
    with open("generated_non_extremist.csv", mode="r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    new_rows = []
    for row in rows:
        text = row[0]
        paraphrased_text = " ".join(get_synonym(w) for w in text.split())
        new_rows.append([paraphrased_text, 0])
    with open("generated_non_extremist.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)

if __name__ == "__main__":
    main()