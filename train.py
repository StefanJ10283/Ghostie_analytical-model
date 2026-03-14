"""
Train a TF-IDF + Logistic Regression sentiment classifier on Financial PhraseBank.

Outputs:
    sentiment_model.pkl
"""

import io
import zipfile
import joblib
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Financial PhraseBank — analyst sentences labeled positive/negative/neutral
# Downloaded as ZIP, parsed from Sentences_AllAgree.txt
print("Loading Financial PhraseBank...")
zip_path = hf_hub_download(
    repo_id="takala/financial_phrasebank",
    filename="data/FinancialPhraseBank-v1.0.zip",
    repo_type="dataset",
)

texts  = []
labels = []
label_map = {"negative": 0, "neutral": 1, "positive": 2}
label_names = ["negative", "neutral", "positive"]

with zipfile.ZipFile(zip_path) as zf:
    # Find the AllAgree file (may be nested in a subfolder inside the zip)
    target = next(n for n in zf.namelist() if "Sentences_AllAgree" in n)
    with zf.open(target) as f:
        for line in io.TextIOWrapper(f, encoding="latin-1"):
            line = line.strip()
            if not line or "@" not in line:
                continue
            # Format: "sentence text@label"
            parts = line.rsplit("@", 1)
            if len(parts) != 2:
                continue
            sentence, label = parts[0].strip(), parts[1].strip().lower()
            if label in label_map:
                texts.append(sentence)
                labels.append(label_map[label])

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)}...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=20000,
        sublinear_tf=True,    # log scaling of term frequency
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",  # handles class imbalance
    )),
])

model.fit(X_train, y_train)

print("\n--- Test Set Performance ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_names))

joblib.dump(model, "sentiment_model.pkl")
print("Model saved to sentiment_model.pkl")
