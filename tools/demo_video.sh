#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

IMAGE="${ROUTER_IMAGE:-lagrangian-router:submission}"
INPUT="${1:-data/materialized/dev/inputs.json}"
OUTPUT_DIR="${2:-/tmp/semantic-router-demo-$(date +%Y%m%d-%H%M%S)}"

fail() {
  printf '\nERROR: %s\n' "$1" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || fail "python3 is required"
command -v docker >/dev/null 2>&1 || fail "docker is required"
[[ -f "$INPUT" ]] || fail "missing $INPUT; materialize the public Dev input before recording"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  fail "Docker is not available without a password prompt; run 'sudo -v' before recording"
fi

"${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 || \
  fail "the pinned ARM64 image is not local; pull it before recording"

mkdir -p "$OUTPUT_DIR"
chmod 0777 "$OUTPUT_DIR"
INPUT_ABS="$(realpath "$INPUT")"
OUTPUT_ABS="$(realpath "$OUTPUT_DIR")"
REPORT="$OUTPUT_ABS/report.json"

EPISODES="$(python3 - "$INPUT_ABS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(len(json.load(handle)["episodes"]))
PY
)"

printf '\033[2J\033[H'
printf '%s\n' '============================================================'
printf '%s\n' '  LSA Lagrangian Router - Official Dev Demonstration'
printf '%s\n' '============================================================'
printf 'Image    : %s\n' "$IMAGE"
printf 'Input    : %s public Dev episodes\n' "$EPISODES"
printf 'Features : 64-dimensional LSA + 46 numeric = 110\n'
printf 'Predict  : per-model score, input tokens, output tokens\n'
printf 'Route    : Lagrangian budget optimization\n'
printf 'Runtime  : linux/arm64, read-only, network disabled\n\n'

for tier in fast balanced premium; do
  printf '[RUN] %-8s ' "$tier"
  start="$(date +%s)"
  "${DOCKER[@]}" run --rm \
    --platform linux/arm64 \
    --cpus 2 \
    --memory 2g \
    --memory-swap 2g \
    --pids-limit 32 \
    --ipc none \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=256m \
    --mount "type=bind,src=$INPUT_ABS,dst=/challenge/input/inputs.json,readonly" \
    --mount "type=bind,src=$OUTPUT_ABS,dst=/challenge/output" \
    "$IMAGE" \
    --input /challenge/input/inputs.json \
    --tier "$tier" \
    --output "/challenge/output/$tier.json"
  elapsed="$(( $(date +%s) - start ))"
  printf 'completed in %ss\n' "$elapsed"
done

PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input "$INPUT_ABS" \
  --outcomes data/dev/outcomes.json \
  --submissions "$OUTPUT_ABS" \
  --policy configs/routing-policy.v1.json \
  --report "$REPORT" >/dev/null

python3 - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

print("\nOFFICIAL SELF-CHECK RESULT")
print("-" * 91)
print(f"{'Tier':<10}{'Quality':>13}{'Cost':>14}{'Ratio':>14}{'Limit':>10}{'Pass':>8}  Model counts")
print("-" * 91)
for tier in ("fast", "balanced", "premium"):
    row = report["tiers"][tier]
    counts = row["model_counts"]
    model_counts = (
        f"light={counts.get('ax31-light', 0)}, "
        f"ax31={counts.get('ax31', 0)}, "
        f"think={counts.get('axk1-think', 0)}"
    )
    passed = "YES" if row["budget_passed"] else "NO"
    print(
        f"{tier.title():<10}{row['quality_score']:>13}{row['total_cost']:>14}"
        f"{row['budget_ratio']:>14}{row['budget_multiplier']:>10}{passed:>8}  {model_counts}"
    )
print("-" * 91)
print(f"Final weighted score: {report['final_score']}")
print(f"Submission files    : {sys.argv[1].rsplit('/', 1)[0]}")
print("\nDemo completed successfully.")
PY
