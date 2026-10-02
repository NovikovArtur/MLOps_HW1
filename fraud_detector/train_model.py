from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from fraud_detector.preprocessing import FEATURE_NAMES, preprocess

DATA = Path("fraud_detector/teta-ml-1-2025/train.csv")
OUTPUT = Path("fraud_detector/models/fraud_logreg.json")


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        count = sum(1 for _ in file) - 1
    features = np.empty((count, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty(count, dtype=np.float32)
    with path.open(newline="", encoding="utf-8") as file:
        for i, row in enumerate(csv.DictReader(file)):
            features[i] = preprocess(row)
            labels[i] = float(row["target"])
    return features, labels


def fit_model(x: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(42)
    validation = rng.random(len(y)) < 0.2
    train = ~validation
    mean = x[train].mean(axis=0, dtype=np.float64)
    scale = x[train].std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    x = ((x - mean) / scale).astype(np.float32)
    x_train, y_train = x[train], y[train]
    x_valid, y_valid = x[validation], y[validation]
    n = len(y_train)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, intercept = parameters[:-1], parameters[-1]
        logits = x_train @ weights + intercept
        probabilities = expit(logits)
        penalty = 0.0002 * np.dot(weights, weights)
        loss = np.logaddexp(0, logits).mean() - np.dot(y_train, logits) / n + penalty
        error = probabilities - y_train
        gradient = np.empty_like(parameters)
        gradient[:-1] = (x_train.T @ error) / n + 0.0004 * weights
        gradient[-1] = error.mean()
        return float(loss), gradient

    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[-1] = float(np.log(y_train.mean() / (1 - y_train.mean())))
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 90})
    print(f"Optimization: {result.message}; iterations={result.nit}")

    scores = expit(x_valid @ result.x[:-1] + result.x[-1])
    order = np.argsort(-scores)
    sorted_labels = y_valid[order]
    true_positives = np.cumsum(sorted_labels)
    predicted_positives = np.arange(1, len(scores) + 1)
    f1 = 2 * true_positives / (predicted_positives + y_valid.sum())
    best = int(np.argmax(f1))
    threshold = float(scores[order[best]])
    print(
        f"Validation: rows={len(y_valid)}, frauds={int(y_valid.sum())}, "
        f"best F1={f1[best]:.4f}, threshold={threshold:.6f}, "
        f"precision={true_positives[best] / (best + 1):.4f}, "
        f"recall={true_positives[best] / y_valid.sum():.4f}"
    )
    return {
        "model_type": "logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "threshold": threshold,
        "training_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "validation_f1": float(f1[best]),
    }


if __name__ == "__main__":
    print(f"Reading {DATA}...")
    x, y = load_data(DATA)
    print(f"Training on {len(y)} transactions; frauds={int(y.sum())}")
    model = fit_model(x, y)
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT}")
