import joblib

model = joblib.load("extremism_tfidf_logreg.joblib")



def classify_text(text, model, threshold):
    prob = model.predict_proba([text])[0,1]
    
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
    "The quick brown fox jumps over the lazy dog. We should destroy all our enemies with nuclear weapons.",
    "We should overthrow the government by force.",
    "Education is the key to a better future for everyone.",
    "Join us in our militant operation to seize power.",
    "We should destroy all our enemies with nuclear weapons.",
    "disarm all nukes now",
    "Lets make some cookies",
    "I love people and want to help them",
    "We must use nuclear weapons to nuke them all.",
    "she is a good person and loves her family",
    "she is a bitch",
    "Bears are amazing creatures that live in the forest.",
]

for txt in examples:
    result = classify_text(txt, model, threshold=0.3645504266543817)
    print(result)
