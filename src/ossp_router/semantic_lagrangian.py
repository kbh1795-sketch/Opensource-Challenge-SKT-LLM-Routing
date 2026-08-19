# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Submission router: frozen LSA semantic embedding + Lagrangian selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import unicodedata
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Optional, Sequence

from .protocol import (
    TIERS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    dumps_json,
    load_bundled_policy,
    load_input,
    load_policy,
    parse_submission,
    submission_to_dict,
)

MODEL_IDS = ("ax31-light", "ax31", "axk1-think")
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
_WORD = re.compile(r"[A-Za-z가-힣]+")
_NUMBER = re.compile(r"(?:\d+(?:[.,]\d+)?)")
_SENTENCE = re.compile(r"[.!?。！？]")
_MATH = re.compile(r"[=+\-*/^∑∫√≈≠≤≥<>]|\\(?:frac|sum|int|sqrt)\b")
_CODE = re.compile(
    r"```|(?:^|\s)(?:def|class|function|SELECT|FROM|import|#include|return|assert)\b|[{};]",
    re.IGNORECASE | re.MULTILINE,
)
_QUESTION = re.compile(
    r"\?|\b(?:what|why|how|which|when|where|who)\b|(?:무엇|왜|어떻게|어느|언제|어디|누구)",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"\b(?:write|create|implement|calculate|solve|explain|summarize|translate|compare|list|prove|derive)\b|"
    r"(?:작성|구현|계산|풀어|설명|요약|번역|비교|나열|증명|유도)",
    re.IGNORECASE,
)
_TASK_PATTERNS = (
    re.compile(r"\b(?:calculate|equation|integer|probability|geometry|algebra|theorem)\b|(?:계산|방정식|정수|확률|기하|대수|정리)", re.I),
    re.compile(r"\b(?:code|program|debug|algorithm|complexity|python|java|sql)\b|(?:코드|프로그램|디버그|알고리즘|복잡도)", re.I),
    re.compile(r"\b(?:reason|analyze|explain why|prove|derive|step by step)\b|(?:추론|분석|이유|증명|유도|단계별)", re.I),
    re.compile(r"\btranslat(?:e|ion)\b|번역", re.I),
    re.compile(r"\bsummari[sz]e\b|요약", re.I),
    re.compile(r"\b(?:story|poem|creative|brainstorm)\b|(?:이야기|시를|창작|아이디어)", re.I),
    re.compile(r"(?:^|\n)\s*(?:[A-D]|[1-4])[.)]\s+|다음 중", re.I),
)
_DIFFICULTY_PATTERNS = (
    re.compile(r"\b(?:must|constraint|exactly|at least|at most|without|except)\b|(?:반드시|제약|정확히|이상|이하|없이|제외)", re.I),
    re.compile(r"\b(?:first|then|finally|step|multi-step)\b|(?:먼저|다음|마지막|단계)", re.I),
    re.compile(r"\b(?:prove|proof|derive|theorem|lemma)\b|(?:증명|유도|정리|보조정리)", re.I),
    re.compile(r"\b(?:compare|trade-?off|pros and cons|difference)\b|(?:비교|장단점|차이|트레이드오프)", re.I),
)
_ARTIFACT = None


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(1.0, denominator)


def _episode_text(episode: Episode) -> tuple[str, int]:
    if episode.prompt is not None:
        return episode.prompt, 1
    assert episode.messages is not None
    return (
        "\n".join(f"<{message.role}>\n{message.content}" for message in episode.messages),
        len(episode.messages),
    )


def _numeric_features(text: str, message_count: int) -> list[float]:
    chars = len(text)
    nonspace = sum(not character.isspace() for character in text)
    words = _WORD.findall(text)
    punctuation = sum(character in string.punctuation for character in text)
    symbols = sum(unicodedata.category(character).startswith("S") for character in text)
    math_count = len(_MATH.findall(text))
    code_count = len(_CODE.findall(text))
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in text)
    latin = sum("a" <= character.lower() <= "z" for character in text)
    cjk = sum("\u4e00" <= character <= "\u9fff" for character in text)
    base = [
        chars,
        len(text.encode("utf-8")),
        nonspace,
        len(words),
        len(_SENTENCE.findall(text)),
        text.count("\n") + 1,
        message_count,
        sum(character.isdigit() for character in text),
        len(_NUMBER.findall(text)),
        punctuation,
        _ratio(punctuation, nonspace),
        symbols,
        _ratio(symbols, nonspace),
        math_count,
        _ratio(math_count, nonspace),
        code_count,
        _ratio(code_count, nonspace),
        _ratio(hangul, nonspace),
        _ratio(latin, nonspace),
        _ratio(cjk, nonspace),
        _ratio(sum(character.isupper() for character in text), nonspace),
        _ratio(sum(len(word) for word in words), len(words)),
        len(_QUESTION.findall(text)),
        len(_INSTRUCTION.findall(text)),
        text.count("?"),
        text.count("!"),
    ]
    base.extend(len(pattern.findall(text)) for pattern in _TASK_PATTERNS)
    lowered_words = [word.lower() for word in words]
    base.extend([
        float(chars >= 500),
        float(chars >= 2_000),
        float(chars >= 8_000),
        float(len(words) >= 350),
        _ratio(len(set(lowered_words)), len(lowered_words)),
        text.count("(") + text.count("[") + text.count("{"),
        max(text.count("("), text.count(")")),
        len(re.findall(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+", text)),
        len(re.findall(r"\b(?:if|else|elif|for|while|try|except|case|when)\b", text, re.I)),
    ])
    base.extend(len(pattern.findall(text)) for pattern in _DIFFICULTY_PATTERNS)
    if len(base) != 46:
        raise AssertionError(f"unexpected numeric feature count: {len(base)}")
    return base


def _load_artifact() -> dict:
    global _ARTIFACT
    if _ARTIFACT is None:
        path = resources.files("ossp_router.resources").joinpath(
            "semantic-lagrangian.v1.json"
        )
        _ARTIFACT = json.loads(path.read_text(encoding="utf-8"))
    return _ARTIFACT


def _semantic_embedding(text: str, artifact: dict) -> list[float]:
    semantic = artifact["semantic_embedding"]
    vocabulary = semantic["vocabulary"]
    tokens = _TOKEN.findall(text.lower())
    counts: Counter[int] = Counter()
    for token in tokens:
        index = vocabulary.get(token)
        if index is not None:
            counts[index] += 1
    for left, right in zip(tokens, tokens[1:]):
        index = vocabulary.get(f"{left} {right}")
        if index is not None:
            counts[index] += 1
    weighted = {
        index: (1.0 + math.log(count)) * semantic["idf"][index]
        for index, count in counts.items()
    }
    norm = math.sqrt(math.fsum(value * value for value in weighted.values()))
    if norm > 0:
        weighted = {index: value / norm for index, value in weighted.items()}
    embedding = [0.0] * semantic["dimensions"]
    components = semantic["components_by_feature"]
    for index, value in weighted.items():
        component = components[index]
        for dimension, coefficient in enumerate(component):
            embedding[dimension] += value * coefficient
    return embedding


def _linear(head: dict, features: list[float]) -> float:
    return head["intercept"] + math.fsum(
        coefficient * value for coefficient, value in zip(head["coefficients"], features)
    )


def _predict_one(text: str, message_count: int, artifact: dict):
    raw = _numeric_features(text, message_count) + _semantic_embedding(text, artifact)
    scaled = [
        (value - mean) / scale
        for value, mean, scale in zip(raw, artifact["feature_mean"], artifact["feature_scale"])
    ]
    scores = []
    outputs = []
    inputs = []
    for model_id in MODEL_IDS:
        scores.append(min(1.0, max(0.0, _linear(artifact["score_heads"][model_id], scaled))))
        output_log = min(
            math.log1p(artifact["output_caps"][model_id]),
            max(0.0, _linear(artifact["output_heads"][model_id], scaled)),
        )
        input_log = min(
            math.log1p(artifact["input_caps"][model_id]),
            max(0.0, _linear(artifact["input_heads"][model_id], scaled)),
        )
        outputs.append(math.expm1(output_log))
        inputs.append(max(1.0, math.expm1(input_log)))
    return scores, outputs, inputs


def _predicted_costs(
    predicted_inputs: list[list[float]],
    predicted_outputs: list[list[float]],
    policy: RoutingPolicy,
    artifact: dict,
) -> list[list[float]]:
    rows = []
    for input_row, output_row in zip(predicted_inputs, predicted_outputs):
        costs = []
        for index, model_id in enumerate(MODEL_IDS):
            rates = policy.models[model_id]
            raw = (
                float(rates.fixed_cost)
                + input_row[index] * float(rates.input_token_rate) / policy.token_unit
                + output_row[index] * float(rates.output_token_rate) / policy.token_unit
            )
            costs.append(raw * artifact["cost_calibration"]["applied_cost_factors"][model_id])
        rows.append(costs)
    return rows


def _lagrangian_select(
    scores: list[list[float]], costs: list[list[float]], budget: float
) -> list[int]:
    cheapest = [min(range(3), key=lambda model: row[model]) for row in costs]
    best = cheapest
    best_score = math.fsum(row[model] for row, model in zip(scores, best))

    def choose(price: float):
        selected = [
            max(range(3), key=lambda model: score_row[model] - price * cost_row[model])
            for score_row, cost_row in zip(scores, costs)
        ]
        total_cost = math.fsum(row[model] for row, model in zip(costs, selected))
        total_score = math.fsum(row[model] for row, model in zip(scores, selected))
        return selected, total_cost, total_score

    unconstrained, unconstrained_cost, unconstrained_score = choose(0.0)
    if unconstrained_cost <= budget:
        return unconstrained
    low, high = 0.0, 1.0
    while choose(high)[1] > budget:
        high *= 2.0
        if high > 1e15:
            raise RuntimeError("failed to bracket a feasible Lagrange multiplier")
    for _ in range(100):
        midpoint = (low + high) / 2.0
        selected, total_cost, total_score = choose(midpoint)
        if total_cost <= budget:
            high = midpoint
            if total_score > best_score:
                best, best_score = selected, total_score
        else:
            low = midpoint
    return best


def make_submission(inputs: InputBatch, policy: RoutingPolicy, tier: str) -> Submission:
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    artifact = _load_artifact()
    if artifact["policy_id"] != policy.policy_id:
        raise ProtocolError("학습 artifact와 비용 정책의 policy_id가 다릅니다.")
    predictions = [_predict_one(*_episode_text(episode), artifact) for episode in inputs.episodes]
    scores = [item[0] for item in predictions]
    outputs = [item[1] for item in predictions]
    input_tokens = [item[2] for item in predictions]
    costs = _predicted_costs(input_tokens, outputs, policy, artifact)
    light_total = math.fsum(row[0] for row in costs)
    multiplier = float(policy.tiers[tier].budget_multiplier)
    headroom = artifact["routing"]["headroom_factor"]
    budget = light_total * (1.0 + (multiplier - 1.0) * headroom)
    selected = _lagrangian_select(scores, costs, budget)
    result = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, MODEL_IDS[model])
            for episode, model in zip(inputs.episodes, selected)
        ),
    )
    return parse_submission(submission_to_dict(result))


def _write_atomic(path: Path, submission: Submission) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            dumps_json(submission_to_dict(submission)), encoding="utf-8"
        )
        temporary.chmod(0o644)
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="router-run")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        _write_atomic(args.output, make_submission(inputs, policy, args.tier))
    except (OSError, ProtocolError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
