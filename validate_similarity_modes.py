#!/usr/bin/env python3
"""
Validate every retrieval space on the reference domains.

For each domain (Iris, UCI Breast Cancer, hardware trojan detection, and
credit-card fraud when ``creditcard.csv`` is available) this script trains a
random forest, a depth-limited decision tree, and an MLP, then explains the
test set in every retrieval space the library offers:

* features (random forest predictions)
* hidden activations (last hidden layer, mean-abs output weighting)
* hidden activations with predicted-class output weighting
* feature/activation hybrid (30% features)
* decision-tree leaf (nearest overflow)
* random-forest proximity

Each space is run three ways: raw distances, ``distance_normalization=
"median_nn"``, and raw distances with ``class_weights="balanced"``. The
neighbors are identical in all three; only the correspondence changes.

Reported per row: model accuracy on the explained samples, mean
correspondence, mean correspondence on correct and incorrect predictions, the
area under the ROC curve of correspondence as a predictor of correctness
(``corr_auc``), the mean distance to the nearest retrieved case, the batch
explanation time per sample, and for each interpretation label (high, medium,
low) how many predictions received it and how many of those were correct.

Usage::

    python validate_similarity_modes.py                      # all domains
    python validate_similarity_modes.py --domains iris breast_cancer
    python validate_similarity_modes.py --max-test 1000 --output results.csv
"""

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_explainer import (  # noqa: E402
    Blend,
    CaseExplainer,
    Features,
    ForestProximity,
    HiddenActivations,
    TreeLeaf,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TROJAN_CSV = os.path.join(HERE, "assets", "data", "hardware_trojan.csv")
FRAUD_CSV = os.path.join(HERE, "creditcard.csv")
DEFAULT_OUTPUT = os.path.join(HERE, "benchmarks", "similarity_validation.csv")
SEED = 42


@dataclass
class Row:
    domain: str
    retrieval_space: str
    normalization: str
    class_weights: str
    n_train: int
    n_test: int
    model: str
    accuracy: float
    mean_correspondence: float
    correspondence_correct: float
    correspondence_incorrect: float
    n_incorrect: int
    corr_auc: float
    mean_nearest_distance: float
    distance_scale: float
    ms_per_sample: float
    n_high: int
    acc_high: float
    n_medium: int
    acc_medium: float
    n_low: int
    acc_low: float


def load_domain(name: str, max_test: Optional[int]):
    """Return ``(X_train, X_test, y_train, y_test)`` for a named domain."""
    if name == "iris":
        X, y = load_iris(return_X_y=True)
    elif name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif name == "hardware_trojan":
        frame = pd.read_csv(TROJAN_CSV)
        X = frame.drop(columns="Trojan").values.astype(float)
        y = frame["Trojan"].values
    elif name == "fraud":
        if not os.path.exists(FRAUD_CSV):
            raise FileNotFoundError(
                f"{FRAUD_CSV} not found; download creditcard.csv from "
                "https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )
        frame = pd.read_csv(FRAUD_CSV)
        X = frame.drop(columns="Class").values.astype(float)
        y = frame["Class"].values
    else:
        raise ValueError(f"Unknown domain {name!r}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    if max_test is not None and len(X_test) > max_test:
        keep = _cap_test_subset(y_test, max_test)
        X_test, y_test = X_test[keep], y_test[keep]
    return X_train, X_test, y_train, y_test


def _cap_test_subset(y_test: np.ndarray, max_test: int) -> np.ndarray:
    """Choose at most ``max_test`` test rows, keeping every minority-class row.

    A uniform random subset of a heavily imbalanced test set contains only a
    handful of positives (about three frauds in 2,000 rows), which makes the
    accuracy and the correspondence AUC meaningless. Rows from every class
    except the largest are kept in full and the remaining budget is filled with
    a random sample of the majority class.
    """
    rng = np.random.RandomState(SEED)
    classes: np.ndarray
    counts: np.ndarray
    classes, counts = np.unique(y_test, return_counts=True)
    majority = classes[int(np.argmax(counts))]
    minority_idx = np.flatnonzero(y_test != majority)
    majority_idx = np.flatnonzero(y_test == majority)
    budget = max(max_test - len(minority_idx), 0)
    if len(minority_idx) >= max_test:
        return np.sort(rng.choice(minority_idx, size=max_test, replace=False))
    fill = rng.choice(majority_idx, size=min(budget, len(majority_idx)), replace=False)
    return np.sort(np.concatenate([minority_idx, fill]))


def fit_models(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    forest.fit(X_train, y_train)
    tree = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(X_train, y_train)
    scaler = StandardScaler().fit(X_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=len(X_train) > 5000,
        random_state=SEED,
    )
    mlp.fit(scaler.transform(X_train), y_train)
    return forest, tree, mlp, scaler


def retrieval_spaces(forest, tree, mlp, scaler):
    hidden = HiddenActivations(model=mlp, input_transform=scaler)
    return [
        ("features", Features(model=forest), "random_forest"),
        ("activations", hidden, "mlp"),
        (
            "activations_predicted_class",
            HiddenActivations(
                model=mlp, input_transform=scaler, output_weighting="predicted_class"
            ),
            "mlp",
        ),
        ("hybrid_features_0.3", Blend(hidden, features=0.3), "mlp"),
        ("tree_leaf", TreeLeaf(model=tree, overflow="nearest"), "decision_tree"),
        ("forest_proximity", ForestProximity(model=forest), "random_forest"),
    ]


#: ``(distance_normalization, class_weights)`` variants run for every space.
VARIANTS: List[Tuple[Optional[str], Optional[str]]] = [
    (None, None),
    ("median_nn", None),
    (None, "balanced"),
]
LABELS = ("high", "medium", "low")


def evaluate(
    domain, space_name, similarity, model_name, normalization, class_weights, data
) -> Row:
    X_train, X_test, y_train, y_test = data
    explainer = CaseExplainer(
        X_train,
        y_train,
        k=5,
        similarity=similarity,
        distance_normalization=normalization,
        class_weights=class_weights,
    )
    start = time.perf_counter()
    explanations = explainer.explain_batch(X_test, y_test=y_test)
    elapsed = time.perf_counter() - start

    correspondence = np.array([e.correspondence for e in explanations])
    correct = np.array([bool(e.is_correct()) for e in explanations])
    nearest = np.array(
        [e.neighbors[0].distance if e.neighbors else np.nan for e in explanations]
    )
    labels = np.array([e.correspondence_interpretation for e in explanations])
    by_label = {}
    for label in LABELS:
        mask = labels == label
        by_label[f"n_{label}"] = int(mask.sum())
        by_label[f"acc_{label}"] = (
            float(correct[mask].mean()) if mask.any() else float("nan")
        )
    incorrect = ~correct
    if 0 < incorrect.sum() < len(correct):
        auc = float(roc_auc_score(correct, correspondence))
    else:
        auc = float("nan")
    return Row(
        domain=domain,
        retrieval_space=space_name,
        normalization=normalization or "raw",
        class_weights=class_weights or "none",
        n_train=len(X_train),
        n_test=len(X_test),
        model=model_name,
        accuracy=float(correct.mean()),
        mean_correspondence=float(correspondence.mean()),
        correspondence_correct=float(correspondence[correct].mean()),
        correspondence_incorrect=(
            float(correspondence[incorrect].mean()) if incorrect.any() else float("nan")
        ),
        n_incorrect=int(incorrect.sum()),
        corr_auc=auc,
        mean_nearest_distance=float(np.nanmean(nearest)),
        distance_scale=float(explainer.distance_scale_),
        ms_per_sample=1000.0 * elapsed / len(X_test),
        **by_label,
    )


def markdown_table(rows: List[Row]) -> str:
    header = (
        "| Domain | Retrieval space | Model | Accuracy | Mean corr. (raw) | "
        "Mean corr. (normalized) | Correct / incorrect (normalized) | "
        "Corr. AUC | ms/sample |"
    )
    lines = [header, "|" + " --- |" * 9]
    by_key = {
        (r.domain, r.retrieval_space, r.normalization): r
        for r in rows
        if r.class_weights == "none"
    }
    seen = []
    for r in rows:
        key = (r.domain, r.retrieval_space)
        if key in seen:
            continue
        seen.append(key)
        raw = by_key[(r.domain, r.retrieval_space, "raw")]
        norm = by_key.get((r.domain, r.retrieval_space, "median_nn"), raw)
        incorrect = (
            f"{norm.correspondence_incorrect:.1%}"
            if np.isfinite(norm.correspondence_incorrect)
            else "n/a"
        )
        auc = f"{norm.corr_auc:.2f}" if np.isfinite(norm.corr_auc) else "n/a"
        lines.append(
            f"| {r.domain} | {r.retrieval_space} | {r.model} | {raw.accuracy:.1%} | "
            f"{raw.mean_correspondence:.1%} | {norm.mean_correspondence:.1%} | "
            f"{norm.correspondence_correct:.1%} / {incorrect} | {auc} | "
            f"{raw.ms_per_sample:.2f} |"
        )
    return "\n".join(lines)


def _label_cell(count: int, accuracy: float) -> str:
    if count == 0:
        return "0"
    return f"{count:,} ({accuracy:.1%} correct)"


def calibration_table(rows: List[Row], space: str = "features") -> str:
    """Label reliability in one retrieval space, unweighted and balanced."""
    header = (
        "| Domain | Class weights | Mean corr. on incorrect | Corr. AUC | "
        "Labeled high | Labeled medium | Labeled low |"
    )
    lines = [header, "|" + " --- |" * 7]
    for r in rows:
        if r.retrieval_space != space or r.normalization != "raw":
            continue
        incorrect = (
            f"{r.correspondence_incorrect:.1%}"
            if np.isfinite(r.correspondence_incorrect)
            else "n/a"
        )
        auc = f"{r.corr_auc:.2f}" if np.isfinite(r.corr_auc) else "n/a"
        lines.append(
            f"| {r.domain} | {r.class_weights} | {incorrect} | {auc} | "
            f"{_label_cell(r.n_high, r.acc_high)} | "
            f"{_label_cell(r.n_medium, r.acc_medium)} | "
            f"{_label_cell(r.n_low, r.acc_low)} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["iris", "breast_cancer", "hardware_trojan", "fraud"],
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=2000,
        help="Cap on explained test samples per domain (default 2000)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows: List[Row] = []
    for domain in args.domains:
        try:
            data = load_domain(domain, args.max_test)
        except FileNotFoundError as exc:
            print(f"[skip] {domain}: {exc}")
            continue
        X_train, X_test, y_train, y_test = data
        print(
            f"\n== {domain}: {len(X_train)} train, {len(X_test)} test, "
            f"{X_train.shape[1]} features =="
        )
        forest, tree, mlp, scaler = fit_models(X_train, y_train)
        for space_name, similarity, model_name in retrieval_spaces(
            forest, tree, mlp, scaler
        ):
            for normalization, class_weights in VARIANTS:
                row = evaluate(
                    domain,
                    space_name,
                    similarity,
                    model_name,
                    normalization,
                    class_weights,
                    data,
                )
                rows.append(row)
                variant = row.normalization
                if row.class_weights != "none":
                    variant = f"{variant}+{row.class_weights}"
                print(
                    f"  {space_name:28s} {variant:12s} acc={row.accuracy:.3f} "
                    f"corr={row.mean_correspondence:.3f} "
                    f"correct={row.correspondence_correct:.3f} "
                    f"incorrect={row.correspondence_incorrect:.3f} "
                    f"auc={row.corr_auc:.3f} d1={row.mean_nearest_distance:.3f} "
                    f"{row.ms_per_sample:.2f} ms"
                )

    if not rows:
        print("No domains were evaluated.")
        return 1
    frame = pd.DataFrame(
        [asdict(r) for r in rows], columns=[f.name for f in fields(Row)]
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.6g")
    print(f"\nWrote {len(rows)} rows to {args.output}\n")
    print(markdown_table(rows))
    print("\nLabel reliability (feature space):\n")
    print(calibration_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
