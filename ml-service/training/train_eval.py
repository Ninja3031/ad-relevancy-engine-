"""
train_eval.py — Embed the corpus and train both model heads.

Head 1: product-centricity (binary, probability = the 0-1 score)
Head 2: category classifier (multiclass over Amazon top-level categories)

Run:  pip install sentence-transformers scikit-learn pandas pyarrow joblib numpy
      python train_eval.py
Runs on CPU; a GPU (free Colab/Kaggle) makes the embedding step ~20x faster.
Output: models/embedder name ref, models/centricity.joblib,
        models/category.joblib, models/labels.txt, eval printout
"""

import os
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
OUT_DIR = "models"


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    products = pd.read_parquet("data/products.parquet")
    negatives = pd.read_parquet("data/negatives.parquet")
    print(f"products={len(products)}  negatives={len(negatives)}")

    model = SentenceTransformer(EMBED_MODEL, device="mps")

    # ---------- embed everything once, reuse for both heads ----------
    print("[+] embedding products ...")
    X_prod = embed(model, products["text"].tolist())
    print("[+] embedding negatives ...")
    X_neg = embed(model, negatives["text"].tolist())

    np.save("data/emb_products.npy", X_prod)   # cache so re-runs are instant
    np.save("data/emb_negatives.npy", X_neg)

    # ---------- Head 1: product-centricity (binary) ----------
    X = np.vstack([X_prod, X_neg])
    y = np.concatenate([np.ones(len(X_prod)), np.zeros(len(X_neg))])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[+] training centricity head ...")
    centricity = LogisticRegression(max_iter=2000, C=1.0)
    centricity.fit(X_tr, y_tr)
    proba = centricity.predict_proba(X_te)[:, 1]
    print(f"    ROC-AUC : {roc_auc_score(y_te, proba):.4f}")
    print(f"    F1@0.5  : {f1_score(y_te, proba >= 0.5):.4f}")
    joblib.dump(centricity, f"{OUT_DIR}/centricity.joblib")

    # ---------- Head 2: category classifier (multiclass) ----------
    labels = sorted(products["category"].unique())
    label_to_id = {c: i for i, c in enumerate(labels)}
    y_cat = products["category"].map(label_to_id).to_numpy()

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_prod, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    print("[+] training category head ...")
    category = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
    category.fit(Xc_tr, yc_tr)
    yc_pred = category.predict(Xc_te)

    print(classification_report(yc_te, yc_pred, target_names=labels, digits=3))
    print(f"    macro-F1: {f1_score(yc_te, yc_pred, average='macro'):.4f}")

    cm = confusion_matrix(yc_te, yc_pred)
    np.savetxt(f"{OUT_DIR}/confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    joblib.dump(category, f"{OUT_DIR}/category.joblib")
    with open(f"{OUT_DIR}/labels.txt", "w") as f:
        f.write("\n".join(labels))
    with open(f"{OUT_DIR}/embedder.txt", "w") as f:
        f.write(EMBED_MODEL)

    print("\nDone. Models saved to ./models/ — these are what FastAPI will load.")


if __name__ == "__main__":
    main()
