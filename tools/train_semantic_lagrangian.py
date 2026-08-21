# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train the frozen LSA representation and per-model Ridge heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ossp_router.semantic_lagrangian import MODEL_IDS, _numeric_features

SEED = 20260819


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(episode: dict) -> tuple[str, int]:
    if "prompt" in episode:
        return episode["prompt"], 1
    messages = episode["messages"]
    return (
        "\n".join(f"<{message['role']}>\n{message['content']}" for message in messages),
        len(messages),
    )


def _costs(inputs: np.ndarray, outputs: np.ndarray, policy: dict) -> np.ndarray:
    result = np.zeros_like(outputs, dtype=np.float64)
    unit = float(policy["token_unit"])
    for column, model_id in enumerate(MODEL_IDS):
        rates = policy["models"][model_id]
        result[:, column] = (
            float(rates["fixed_cost"])
            + inputs[:, column] * float(rates["input_token_rate"]) / unit
            + outputs[:, column] * float(rates["output_token_rate"]) / unit
        )
    return result


def _head(estimator: Ridge) -> dict:
    return {
        "intercept": float(estimator.intercept_),
        "coefficients": [float(value) for value in estimator.coef_],
    }


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator)
        if denominator > 0
        else 0.0,
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--embedding-dimensions", type=int, default=64)
    args = parser.parse_args()

    inputs_json = _read(args.input)
    outcomes_json = _read(args.outcomes)
    policy = _read(args.policy)
    if inputs_json["split"] != "train" or outcomes_json["split"] != "train":
        raise ValueError("Only the official Train split may fit the artifact")
    outcome_by_id = {
        episode["episode_id"]: episode for episode in outcomes_json["episodes"]
    }
    texts = []
    message_counts = []
    score_rows = []
    input_rows = []
    output_rows = []
    for episode in inputs_json["episodes"]:
        text, count = _text(episode)
        models = outcome_by_id[episode["episode_id"]]["models"]
        texts.append(text)
        message_counts.append(count)
        score_rows.append([float(models[model]["score"]) for model in MODEL_IDS])
        input_rows.append([float(models[model]["input_tokens"]) for model in MODEL_IDS])
        output_rows.append([float(models[model]["output_tokens"]) for model in MODEL_IDS])
    scores = np.asarray(score_rows, dtype=np.float64)
    input_tokens = np.asarray(input_rows, dtype=np.float64)
    output_tokens = np.asarray(output_rows, dtype=np.float64)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )
    tfidf = vectorizer.fit_transform(texts)
    dimensions = min(
        args.embedding_dimensions, tfidf.shape[0] - 1, tfidf.shape[1] - 1
    )
    svd = TruncatedSVD(n_components=dimensions, n_iter=7, random_state=SEED)
    embeddings = svd.fit_transform(tfidf)
    numeric = np.asarray(
        [_numeric_features(text, count) for text, count in zip(texts, message_counts)],
        dtype=np.float64,
    )
    raw_features = np.hstack([numeric, embeddings])
    scaler = StandardScaler().fit(raw_features)
    features = scaler.transform(raw_features)

    score_heads = {}
    output_heads = {}
    input_heads = {}
    output_caps = {}
    input_caps = {}
    for column, model_id in enumerate(MODEL_IDS):
        score_head = Ridge(alpha=10.0, solver="lsqr").fit(features, scores[:, column])
        output_head = Ridge(alpha=10.0, solver="lsqr").fit(
            features, np.log1p(output_tokens[:, column])
        )
        input_head = Ridge(alpha=10.0, solver="lsqr").fit(
            features, np.log1p(input_tokens[:, column])
        )
        score_heads[model_id] = _head(score_head)
        output_heads[model_id] = _head(output_head)
        input_heads[model_id] = _head(input_head)
        output_caps[model_id] = float(output_tokens[:, column].max() * 2.0)
        input_caps[model_id] = float(input_tokens[:, column].max() * 2.0)
    # Train-only OOF aggregate-cost calibration. The semantic vocabulary and
    # projection are unsupervised and fixed on public Train; only supervised
    # Ridge heads are cross-fitted here.
    oof_scores = np.zeros_like(scores)
    oof_outputs = np.zeros_like(output_tokens)
    oof_inputs = np.zeros_like(input_tokens)
    fold_ids = np.zeros(len(texts), dtype=np.int64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_index, validation_index) in enumerate(splitter.split(features)):
        for column in range(len(MODEL_IDS)):
            score_head = Ridge(alpha=10.0, solver="lsqr").fit(
                features[fit_index], scores[fit_index, column]
            )
            output_head = Ridge(alpha=10.0, solver="lsqr").fit(
                features[fit_index], np.log1p(output_tokens[fit_index, column])
            )
            input_head = Ridge(alpha=10.0, solver="lsqr").fit(
                features[fit_index], np.log1p(input_tokens[fit_index, column])
            )
            output_cap = output_tokens[fit_index, column].max() * 2.0
            input_cap = input_tokens[fit_index, column].max() * 2.0
            oof_scores[validation_index, column] = np.clip(
                score_head.predict(features[validation_index]), 0.0, 1.0
            )
            oof_outputs[validation_index, column] = np.expm1(np.clip(
                output_head.predict(features[validation_index]), 0.0, math.log1p(output_cap)
            ))
            oof_inputs[validation_index, column] = np.expm1(np.clip(
                input_head.predict(features[validation_index]), 0.0, math.log1p(input_cap)
            ))
        fold_ids[validation_index] = fold
    predicted_cost = _costs(oof_inputs, oof_outputs, policy)
    actual_cost = _costs(input_tokens, output_tokens, policy)
    aggregate_ratios = actual_cost.sum(axis=0) / predicted_cost.sum(axis=0)
    fold_ratios = {}
    applied_factors = {}
    for column, model_id in enumerate(MODEL_IDS):
        values = []
        for fold in range(5):
            mask = fold_ids == fold
            values.append(float(actual_cost[mask, column].sum() / predicted_cost[mask, column].sum()))
        fold_ratios[model_id] = values
        applied_factors[model_id] = max(1.0, max(values))
    applied_factors[MODEL_IDS[0]] = float(aggregate_ratios[0])
    headroom_factor = min(
        1.0, float(aggregate_ratios[0]) / max(fold_ratios[MODEL_IDS[0]])
    )

    vocabulary = vectorizer.vocabulary_
    components_by_feature = svd.components_.T
    artifact = {
        "schema_version": 1,
        "artifact_id": "semantic-lagrangian-lsa-v1",
        "policy_id": policy["policy_id"],
        "model_ids": list(MODEL_IDS),
        "seed": SEED,
        "training": {
            "split": "train",
            "episodes": len(texts),
            "input_sha256": _sha256(args.input),
            "outcomes_sha256": _sha256(args.outcomes),
            "policy_sha256": _sha256(args.policy),
        },
        "semantic_embedding": {
            "method": "TF-IDF word unigram+bigram followed by TruncatedSVD (LSA)",
            "dimensions": dimensions,
            "token_pattern": r"(?u)\b\w\w+\b",
            "lowercase": True,
            "sublinear_tf": True,
            "norm": "l2",
            "vocabulary": {
                term: int(index) for term, index in vocabulary.items()
            },
            "idf": [float(value) for value in vectorizer.idf_],
            "components_by_feature": [
                [float(value) for value in row] for row in components_by_feature
            ],
        },
        "numeric_feature_count": int(numeric.shape[1]),
        "feature_mean": [float(value) for value in scaler.mean_],
        "feature_scale": [float(value) for value in scaler.scale_],
        "score_heads": score_heads,
        "output_heads": output_heads,
        "input_heads": input_heads,
        "output_caps": output_caps,
        "input_caps": input_caps,
        "cost_calibration": {
            "method": "5-fold Train-only OOF aggregate cost ratios",
            "aggregate_cost_ratios": {
                model_id: float(aggregate_ratios[index])
                for index, model_id in enumerate(MODEL_IDS)
            },
            "fold_cost_ratios": fold_ratios,
            "applied_cost_factors": applied_factors,
            "headroom_factor": headroom_factor,
        },
        "routing": {
            "method": "Lagrangian argmax(score - lambda * cost)",
            "headroom_factor": 0.85,
            "headroom_policy": "fixed 85% of incremental tier budget",
        },
    }
    _atomic_json(args.artifact, artifact)
    report = {
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": _sha256(args.artifact),
        "artifact_bytes": args.artifact.stat().st_size,
        "episodes": len(texts),
        "tfidf_features": len(vocabulary),
        "semantic_dimensions": dimensions,
        "numeric_dimensions": int(numeric.shape[1]),
        "final_dimensions": int(features.shape[1]),
        "cost_calibration": artifact["cost_calibration"],
        "routing": artifact["routing"],
        "train_oof_regression_metrics": {
            model_id: {
                "score": _regression_metrics(scores[:, index], oof_scores[:, index]),
                "output_tokens": _regression_metrics(
                    output_tokens[:, index], oof_outputs[:, index]
                ),
                "input_tokens": _regression_metrics(
                    input_tokens[:, index], oof_inputs[:, index]
                ),
            }
            for index, model_id in enumerate(MODEL_IDS)
        },
    }
    _atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
