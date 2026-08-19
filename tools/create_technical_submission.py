# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Create the exact six-field technical submission after publishing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from validate_technical_submission import (
    DEFAULT_SCHEMA,
    _load_json,
    validate_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--primary-license", default="Apache-2.0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission-ossp-skt.json",
    )
    args = parser.parse_args()
    value = {
        "schema_version": 1,
        "challenge_id": "ossp-2026-llm-router-challenge",
        "repository_url": args.repository_url,
        "commit_sha": args.commit_sha,
        "image_digest": args.image_digest,
        "primary_license": args.primary_license,
    }
    validated = validate_submission(value, _load_json(DEFAULT_SCHEMA))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(args.output))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    print(f"OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
