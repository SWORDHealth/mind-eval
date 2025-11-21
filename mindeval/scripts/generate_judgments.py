from typing import Any

from jsonargparse import CLI
from mindeval.inference import InferenceEngine
from mindeval.judge_prompts import JUDGE_PROMPT_TEMPLATE_VERSION_DICT
from mindeval.utils import (
    load_jsonl,
    messages_to_convo_str,
    parse_judge_scores,
    save_to_jsonl,
)
from tqdm import tqdm


def main(
    interactions_path: str,
    judge_template_version: str,
    judge_model_api_params: dict[str, Any],
    max_workers: int,
    output_path: str,
):
    # load interactions
    all_packages = load_jsonl(interactions_path)
    all_profiles = [p["member_profile"] for p in all_packages]
    all_interactions = [p["interaction"] for p in all_packages]
    n_interactions = len(all_interactions)
    # load judge prompt (user prompt!)
    judge_prompt_template = JUDGE_PROMPT_TEMPLATE_VERSION_DICT[judge_template_version]
    # instantiate judge model
    judge_model = InferenceEngine(judge_model_api_params)
    # run inference
    all_unparsed_judgments = []
    all_judge_thinking_traces = []
    for batch_start in tqdm(
        range(0, n_interactions, max_workers),
        total=(n_interactions + max_workers - 1) // max_workers,
        desc="Judging interactions",
    ):
        in_profiles = all_profiles[
            batch_start : min(batch_start + max_workers, n_interactions)
        ]
        in_interactions = all_interactions[
            batch_start : min(batch_start + max_workers, n_interactions)
        ]
        messages = []
        for profile, interaction in zip(in_profiles, in_interactions):
            convo_str = messages_to_convo_str(interaction)
            user_prompt = judge_prompt_template.substitute(
                conversation_str=convo_str,
                **profile,
            )
            messages.append([{"role": "user", "content": user_prompt}])
        judgments, thinking = judge_model.batch_generate_with_thinking(messages)
        all_unparsed_judgments.extend(judgments)
        all_judge_thinking_traces.extend(thinking)
    # final save
    outputs = [
        {
            "parsed_judgment": parse_judge_scores(u)[0],
            "unparsed_judgment": u,
            "thinking_trace": t,
        }
        for u, t in zip(all_unparsed_judgments, all_judge_thinking_traces)
    ]
    save_to_jsonl(outputs, output_path)


if __name__ == "__main__":
    CLI([main], as_positional=False)
