import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)

# load the dataset
df = pd.read_csv("combined_dataset.csv").dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

# split intpo train, val, test
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)

X_train, y_train = train_df["text"].values, train_df["label"].values
X_val,   y_val   = val_df["text"].values,   val_df["label"].values
X_test,  y_test  = test_df["text"].values,  test_df["label"].values

# pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,3), min_df=1, max_df=0.9, sublinear_tf=True)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=200, n_jobs=-1))
])

# train
pipe.fit(X_train, y_train)

# validation and choosing threshold
probs_val = pipe.predict_proba(X_val)[:,1]
prec, rec, thr = precision_recall_curve(y_val, probs_val)
ix = np.where(prec >= 0.90)[0]
best_thr = thr[ix[0]-1] if len(ix) else 0.5
print("ROC-AUC (val):", roc_auc_score(y_val, probs_val))
print("PR-AUC  (val):", average_precision_score(y_val, probs_val))
print("Chosen threshold:", best_thr)

# testing
probs_test = pipe.predict_proba(X_test)[:,1]
preds_test = (probs_test >= best_thr).astype(int)
print("\nClassification report (test):")
print(classification_report(y_test, preds_test, digits=3))
print("ROC-AUC (test):", roc_auc_score(y_test, probs_test))
print("PR-AUC  (test):", average_precision_score(y_test, probs_test))
print("Confusion matrix (test):\n", confusion_matrix(y_test, preds_test))

# Save model
joblib.dump(pipe, "extremism_tfidf_logreg.joblib")
