OUTPUT_DIR=$1
JUDGE_VERSION=${2:-mindeval2}

echo Running interactions into: $OUTPUT_DIR

python mindeval/scripts/generate_interactions.py \
    --output_dir $OUTPUT_DIR

echo Running judgments \($JUDGE_VERSION\) for: $OUTPUT_DIR

python mindeval/scripts/generate_judgments.py \
    --output_dir $OUTPUT_DIR \
    --judge_version $JUDGE_VERSION

echo Printing summary of results

python -c "import json; print(json.load(open('$OUTPUT_DIR/judge/$JUDGE_VERSION/summary.json')))"
