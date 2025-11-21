CLINICIAN_MODEL_API_PARAMS=$1

echo Running interactions with clinician model: $CLINICIAN_MODEL_API_PARAMS

python mindeval/scripts/generate_interactions.py \
    --profiles_path data/profiles.jsonl \
    --clinician_system_template_version custom \
    --clinician_model_api_params $CLINICIAN_MODEL_API_PARAMS \
    --member_system_template_version v0_2 \
    --member_model_api_params "{'model':'PROVIDER/claude-haiku-4-5@20251001','max_completion_tokens':16000,'reasoning_effort':'high','api_base':'API_BASE'}" \
    --n_turns 10 \
    --max_workers 50 \
    --output_path interactions.jsonl

echo Running judgements with clinician model: $CLINICIAN_MODEL_API_PARAMS

python mindeval/scripts/generate_judgments.py \
    --interactions_path interactions.jsonl \
    --judge_template_version v0_1 \
    --judge_model_api_params "{'model':'PROVIDER/claude-sonnet-4-5@20250929','max_completion_tokens':16000,'reasoning_effort':'high','api_base':'API_BASE'}" \
    --max_workers 50 \
    --output_path judgments.jsonl

echo Printing summary of results

python -c "import pandas as pd; from mindeval.utils import load_jsonl; full_judgments = load_jsonl('judgments.jsonl'); float_judgments = [j['parsed_judgment'] for j in full_judgments]; df = pd.DataFrame(float_judgments); print(df.mean())" > results.txt