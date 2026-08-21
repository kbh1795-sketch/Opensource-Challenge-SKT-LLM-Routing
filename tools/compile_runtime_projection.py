# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Add the algebraically fused runtime projection to a trained artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ossp_router.semantic_lagrangian import _build_runtime_projection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    artifact["runtime_projection"] = _build_runtime_projection(artifact)
    temporary = args.artifact.with_name(f".{args.artifact.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(args.artifact))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
