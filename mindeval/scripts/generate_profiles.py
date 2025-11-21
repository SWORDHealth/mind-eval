from pathlib import Path
from string import Template
from typing import Any

from jsonargparse import CLI
from tqdm import tqdm

from mindeval.inference import InferenceEngine
from mindeval.prompts import PROFILE_GENERATION_VERSION_TO_TEMPLATE
from mindeval.utils import save_to_jsonl
from mindeval.variables import sample_one_profile


def generate_narrative_batch(
    profiles: list[dict], model: InferenceEngine, profile_generation_template: Template
) -> tuple[list[str], list[str]]:
    messages = [
        [
            {
                "role": "user",
                "content": profile_generation_template.substitute(**p),
            }
        ]
        for p in profiles
    ]
    narratives, thinking = model.batch_generate_with_thinking(messages)
    return narratives, thinking


def main(
    n_profiles: int,
    output_path: str,
    api_params: dict[str, Any],
    profile_generation_template_version: str,
    max_workers: int = 80,
    base_seed: int = 42,
):
    api_params["max_workers"] = max_workers
    model = InferenceEngine(
        api_params=api_params,
    )
    profile_generation_template = PROFILE_GENERATION_VERSION_TO_TEMPLATE[
        profile_generation_template_version
    ]
    output_path = Path(output_path)
    backup_path = output_path.parent / f"part.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # run inference
    out_profiles = []
    print(f"Using model with params: {model.api_params}")
    for batch_start in tqdm(
        range(0, n_profiles, max_workers),
        total=(n_profiles + max_workers - 1) // max_workers,
        desc="Generating profiles",
    ):
        profiles = [
            sample_one_profile(base_seed * i)
            for i in range(batch_start, min(batch_start + max_workers, n_profiles))
        ]
        narratives, thinking = generate_narrative_batch(
            profiles, model, profile_generation_template
        )
        out_profiles.extend(
            [
                {
                    "member_attributes": p,
                    "member_narrative": n,
                    "member_narrative_thinking_trace": t,
                    "model_params": model.api_params,
                }
                for p, n, t in zip(profiles, narratives, thinking)
            ]
        )
        # save backup
        save_to_jsonl(out_profiles, backup_path)
    # final save
    save_to_jsonl(out_profiles, output_path)


if __name__ == "__main__":
    CLI([main], as_positional=False)
