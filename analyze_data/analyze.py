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

def analyze_data(data_str):
    return find_extreme_parts(data_str)

