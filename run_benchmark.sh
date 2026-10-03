#!/usr/bin/env bash
# Runs interactions then judgments for one counselor, end to end. Exit codes match the two scripts':
# 0 finished, 1 partial (some members/units still pending), 2 refused (bad input, mismatched resume).
# A refused interactions pass (2) is fatal: the run directory holds nothing worth judging, so we stop
# before spending judge calls on it. A partial pass (1) still judges whatever finished, but the script
# keeps propagating 1 unless judging itself fails worse.
set -euo pipefail

OUTPUT_DIR=$1
JUDGE_VERSION=${2:-mindeval2}

echo "Running interactions into: $OUTPUT_DIR"

interactions_status=0
python mindeval/scripts/generate_interactions.py \
    --output_dir "$OUTPUT_DIR" || interactions_status=$?

if [ "$interactions_status" -eq 2 ]; then
    echo "generate_interactions.py refused (exit 2); not judging" >&2
    exit "$interactions_status"
elif [ "$interactions_status" -ne 0 ] && [ "$interactions_status" -ne 1 ]; then
    exit "$interactions_status"
fi

echo "Running judgments ($JUDGE_VERSION) for: $OUTPUT_DIR"

judge_status=0
python mindeval/scripts/generate_judgments.py \
    --output_dir "$OUTPUT_DIR" \
    --judge_version "$JUDGE_VERSION" || judge_status=$?

if [ "$judge_status" -ne 0 ]; then
    exit "$judge_status"
fi

echo "Printing summary of results"

cat "$OUTPUT_DIR/judge/$JUDGE_VERSION/summary.json"

exit "$interactions_status"
