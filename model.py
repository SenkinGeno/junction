import joblib

# Load trained pipeline
model = joblib.load("extremism_tfidf_logreg.joblib")



def classify_text(text, model, threshold):
    # Get probability of extremist (class 1)
    prob = model.predict_proba([text])[0,1]
    
    # Apply threshold
    label = 1 if prob >= threshold else 0
    
    return {
        "text": text,
        "probability_extremist": prob,
        "predicted_label": label
    }


examples = [
    "We must spread love and peace among all people.",
    "Lets bomb japan once again!",
    "We hate black people",
    "Lets bomb asians",
    "The quick brown fox jumps over the lazy dog.",
    "We should overthrow the government by force.",
    "Education is the key to a better future for everyone.",
    "Join us in our militant operation to seize power.",
    "We should destroy all our enemies with nuclear weapons.",
    "disarm all nukes now",
]

for txt in examples:
    result = classify_text(txt, model, threshold=0.5)
    print(result)
