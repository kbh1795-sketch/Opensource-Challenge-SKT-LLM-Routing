# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import tempfile
import unittest

from ossp_router.protocol import (
    load_bundled_policy,
    load_input,
    load_submission,
    parse_input,
)
from ossp_router.semantic_lagrangian import (
    _episode_text,
    _load_artifact,
    _numeric_features,
    _predict_one,
    _predict_one_reference,
    main,
    make_submission,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _batch(episode_ids=("one", "two", "three"), *, reverse=False):
    values = [
        {"episode_id": episode_ids[0], "prompt": "한국어 문장을 짧게 요약해 주세요."},
        {
            "episode_id": episode_ids[1],
            "prompt": "Prove x^2 + 2*x + 1 = (x+1)^2 step by step.",
        },
        {
            "episode_id": episode_ids[2],
            "messages": [
                {"role": "system", "content": "Write valid Python."},
                {"role": "user", "content": "Implement an O(n) algorithm."},
            ],
        },
    ]
    if reverse:
        values.reverse()
    return parse_input(
        {
            "schema_version": 1,
            "challenge_id": "ossp-2026-llm-router-challenge",
            "split": "synthetic",
            "episodes": values,
        }
    )


def _by_content(batch, submission):
    selected = {item.episode_id: item.model_id for item in submission.decisions}
    return {
        _episode_text(episode)[0]: selected[episode.episode_id]
        for episode in batch.episodes
    }


class SemanticLagrangianTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_bundled_policy()

    def test_artifact_dimensions_and_train_only_provenance(self) -> None:
        artifact = _load_artifact()
        self.assertEqual("train", artifact["training"]["split"])
        self.assertEqual(64, artifact["semantic_embedding"]["dimensions"])
        self.assertEqual(46, artifact["numeric_feature_count"])
        self.assertEqual(110, len(artifact["feature_mean"]))
        self.assertEqual(0.85, artifact["routing"]["headroom_factor"])
        self.assertEqual(46, len(_numeric_features("Solve 2+2.", 1)))

    def test_fused_runtime_projection_matches_reference_predictions(self) -> None:
        artifact = _load_artifact()
        self.assertIn("runtime_projection", artifact)
        for episode in _batch().episodes:
            text, message_count = _episode_text(episode)
            optimized = _predict_one(text, message_count, artifact)
            reference = _predict_one_reference(text, message_count, artifact)
            for optimized_values, reference_values in zip(optimized, reference):
                for optimized_value, reference_value in zip(
                    optimized_values, reference_values
                ):
                    self.assertAlmostEqual(
                        optimized_value, reference_value, delta=1e-9
                    )

    def test_ids_and_order_do_not_change_content_decisions(self) -> None:
        original = _batch()
        changed = _batch(("opaque-a", "opaque-b", "opaque-c"))
        reordered = _batch(reverse=True)
        for tier in ("fast", "balanced", "premium"):
            expected = _by_content(
                original, make_submission(original, self.policy, tier)
            )
            self.assertEqual(
                expected,
                _by_content(changed, make_submission(changed, self.policy, tier)),
            )
            self.assertEqual(
                expected,
                _by_content(reordered, make_submission(reordered, self.policy, tier)),
            )

    def test_cli_is_byte_deterministic_and_writes_valid_submission(self) -> None:
        input_path = ROOT / "data/toy/inputs.json"
        with tempfile.TemporaryDirectory() as temporary:
            paths = [pathlib.Path(temporary) / f"out-{index}.json" for index in range(2)]
            for path in paths:
                self.assertEqual(
                    0,
                    main(
                        [
                            "--input",
                            str(input_path),
                            "--tier",
                            "balanced",
                            "--output",
                            str(path),
                        ]
                    ),
                )
                submission = load_submission(path)
                self.assertEqual("balanced", submission.tier)
                self.assertEqual(len(load_input(input_path).episodes), len(submission.decisions))
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())


if __name__ == "__main__":
    unittest.main()
