"""Is a 0.5B LLM worth it here? A TF-IDF + logistic regression baseline on the exact same split.

Run: .venv/bin/python analysis/baseline_tfidf.py
"""

import random
import time
import zipfile

from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Same split as train.py: seed 42, shuffle, first 1500 train, next 400 eval.
random.seed(42)
zip_path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
lines = zipfile.ZipFile(zip_path).read("FinancialPhraseBank-v1.0/Sentences_50Agree.txt").decode("latin-1").splitlines()
labels = ["negative", "neutral", "positive"]
examples = []
for line in lines:
    if not line.strip():
        continue
    text, label = line.rsplit("@", 1)
    examples.append((text.strip(), labels.index(label.strip())))
random.shuffle(examples)
train, test = examples[:1500], examples[1500:1900]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
x_train = vectorizer.fit_transform([t for t, _ in train])
x_test = vectorizer.transform([t for t, _ in test])
classifier = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
start = time.perf_counter()
classifier.fit(x_train, [l for _, l in train])
fit_seconds = time.perf_counter() - start
preds = classifier.predict(x_test)
start = time.perf_counter()
for text, _ in test:
    classifier.predict(vectorizer.transform([text]))
latency_ms = 1000 * (time.perf_counter() - start) / len(test)
y = [l for _, l in test]
print(f"majority class share: {max(y.count(c) for c in range(3)) / len(y):.3f}")
print(f"tfidf+logreg: acc {accuracy_score(y, preds):.3f}  F1 {f1_score(y, preds, average='macro'):.3f}  "
      f"fit {fit_seconds:.1f}s  latency {latency_ms:.2f} ms  params {x_train.shape[1] * 3:,}")
