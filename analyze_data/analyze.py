import joblib

model = joblib.load("extremism_tfidf_logreg.joblib")
threshold = 0.3645504266543817

def classify_text(text):
    print("Classifying:", text)
    prob = model.predict_proba([text])[0,1]
    
    label = 1 if prob >= threshold else 0
    print("Probability extremist:", prob, "Predicted label:", label)
    return label



def find_extreme_parts(data_str, window_size=10, overlap=5):
    extreme_parts = []
    if len(data_str.split()) < window_size:
        if classify_text(data_str) == 1:
            print("appending")
            extreme_parts.append(data_str)
        return extreme_parts
    counter = window_size
    while counter < len(data_str.split()):
        window_str = " ".join(data_str.split()[counter-window_size:counter])
        if classify_text(window_str) == 1:
            print("appending")
            extreme_parts.append(window_str)
        counter += overlap
    remain = len(data_str.split()) - counter
    if remain > 0:
        window_str = " ".join(data_str.split()[-window_size:])
        if classify_text(window_str) == 1:
            extreme_parts.append(window_str)

    return extreme_parts

def find_extreme_parts_with_timestamps(words, window_size=10, overlap=5):
    extreme_parts = []
    if len(words) < window_size:
        if classify_text(words_to_text(words)) == 1:
            print("appending")
            extreme_parts = words
        return extreme_parts
    counter = window_size
    while counter < len(words):
        window_str = words_to_text(words, counter-window_size, counter)
        if classify_text(window_str) == 1:
            print("appending")
            extreme_parts = extreme_parts + (words[counter-window_size:counter])
        counter += overlap
    remain = len(words) - counter
    if remain > 0:
        window_str = words_to_text(words, len(words)-window_size, len(words))
        if classify_text(window_str) == 1:
            extreme_parts = extreme_parts + (words[-window_size:])

    return extreme_parts

import re

def find_extreme_sentences_with_timestamps(words):
    extreme_parts = []

    def words_to_text(word_list):
        return ''.join([w['word'] for w in word_list])

    sentences = []
    current_sentence = []
    sentence_end_re = re.compile(r'[.!?]')

    for w in words:
        current_sentence.append(w)
        if sentence_end_re.search(w['word']):
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    for sentence in sentences:
        text = words_to_text(sentence)
        if classify_text(text) == 1:
            print("appending")
            extreme_parts.extend(sentence)

    return extreme_parts



def words_to_text(words, start=0, end=0):
    if end == 0:
        end = len(words)
    return " ".join([w["word"] for w in words[start:end]])
    

def extract_timestamps(extreme_parts):
    if len(extreme_parts) == 0:
        return []
    timestamps = []
    prev_id = -2
    for part in extreme_parts:
        if part['i'] != prev_id + 1:
            timestamps.append(part['start'])
        else:
            pass
        prev_id = part['i']
    return timestamps

def analyze_data(data):
    # extreme_parts = find_extreme_parts_with_timestamps(data)
    extreme_parts = find_extreme_sentences_with_timestamps(data)
    return extract_timestamps(extreme_parts)


# print(analyze_data([{'i': 0, 'word': ' The', 'start': 0.8600000000000003, 'end': 1.46}, {'i': 1, 'word': ' stale', 'start': 1.46, 'end': 1.84}, {'i': 2, 'word': ' smell', 'start': 1.84, 'end': 2.2}, {'i': 3, 'word': ' of', 'start': 2.2, 'end': 2.44}, {'i': 4, 'word': ' old', 'start': 2.44, 'end': 2.72}, {'i': 5, 'word': ' beer', 'start': 2.72, 'end': 2.96}, {'i': 6, 'word': ' lingers.', 'start': 2.96, 'end': 3.64}, {'i': 7, 'word': ' It', 'start': 4.26, 'end': 4.5}, {'i': 8, 'word': ' takes', 'start': 4.5, 'end': 4.82}, {'i': 9, 'word': ' heat', 'start': 4.82, 'end': 5.12}, {'i': 10, 'word': ' to', 'start': 5.12, 'end': 5.34}, {'i': 11, 'word': ' bring', 'start': 5.34, 'end': 5.54}, {'i': 12, 'word': ' out', 'start': 5.54, 'end': 5.74}, {'i': 13, 'word': ' the', 'start': 5.74, 'end': 5.9}, {'i': 14, 'word': ' odor.', 'start': 5.9, 'end': 6.16}, {'i': 15, 'word': ' A', 'start': 7.02, 'end': 7.12}, {'i': 16, 'word': ' cold', 'start': 7.12, 'end': 7.42}, {'i': 17, 'word': ' dip', 'start': 7.42, 'end': 7.78}, {'i': 18, 'word': ' restores', 'start': 7.78, 'end': 8.4}, {'i': 19, 'word': ' health', 'start': 8.4, 'end': 8.72}, {'i': 20, 'word': ' and', 'start': 8.72, 'end': 8.98}, {'i': 21, 'word': ' zest.', 'start': 8.98, 'end': 9.2}, {'i': 22, 'word': ' A', 'start': 9.96, 'end': 10.1}, {'i': 23, 'word': ' salt', 'start': 10.1, 'end': 10.4}, {'i': 24, 'word': ' pickle', 'start': 10.4, 'end': 10.8}, {'i': 25, 'word': ' tastes', 'start': 10.8, 'end': 11.2}, {'i': 26, 'word': ' fine', 'start': 11.2, 'end': 11.54}, {'i': 27, 'word': ' with', 'start': 11.54, 'end': 11.76}, {'i': 28, 'word': ' ham.', 'start': 11.76, 'end': 12.0}, {'i': 29, 'word': ' Tacos', 'start': 12.56, 'end': 12.98}, {'i': 30, 'word': ' al', 'start': 12.98, 'end': 13.14}, {'i': 31, 'word': ' pastor', 'start': 13.14, 'end': 13.42}, {'i': 32, 'word': ' are', 'start': 13.42, 'end': 13.78}, {'i': 33, 'word': ' my', 'start': 13.78, 'end': 13.94}, {'i': 34, 'word': ' favorite.', 'start': 13.94, 'end': 14.26}, {'i': 35, 'word': ' A', 'start': 15.1, 'end': 15.22}, {'i': 36, 'word': ' zestful', 'start': 15.22, 'end': 15.68}, {'i': 37, 'word': ' food', 'start': 15.68, 'end': 16.0}, {'i': 38, 'word': ' is', 'start': 16.0, 'end': 16.38}, {'i': 39, 'word': ' the', 'start': 16.38, 'end': 16.54}, {'i': 40, 'word': ' hot', 'start': 16.54, 'end': 16.78}, {'i': 41, 'word': ' cross', 'start': 16.78, 'end': 17.1}, {'i': 42, 'word': ' bun.', 'start': 17.1, 'end': 17.44}, {'i': 43, 'word': ' We', 'start': 18.32, 'end': 18.52}, {'i': 44, 'word': ' should', 'start': 18.52, 'end': 18.68}, {'i': 45, 'word': ' get', 'start': 18.68, 'end': 18.86}, {'i': 46, 'word': ' rid', 'start': 18.86, 'end': 19.08}, {'i': 47, 'word': ' of', 'start': 19.08, 'end': 19.24}, {'i': 48, 'word': ' people', 'start': 19.24, 'end': 19.54}, {'i': 49, 'word': ' with', 'start': 19.54, 'end': 19.82}, {'i': 50, 'word': ' Down', 'start': 19.82, 'end': 20.08}, {'i': 51, 'word': ' syndrome.', 'start': 20.08, 'end': 20.54}]))