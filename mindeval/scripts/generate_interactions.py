from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template
from typing import Any

from jsonargparse import CLI
from tqdm import tqdm

from mindeval.inference import InferenceEngine
from mindeval.prompts import (
    INTERACTION_CLINICIAN_VERSION_DICT,
    INTERACTION_MEMBER_VERSION_DICT,
)
from mindeval.utils import load_jsonl, save_to_jsonl


def run_interactions(
    profile: dict,
    clinician_system_prompt_template: Template,
    clinician_model: InferenceEngine,
    member_system_prompt_template: Template,
    member_model: InferenceEngine,
    n_turns: int,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Run a multi-turn interaction between clinician and member models given a profile."""
    # prepare initial messages
    clinician_messages = [
        {
            "role": "system",
            "content": clinician_system_prompt_template.substitute(**profile),
        },
        {
            "role": "user",
            "content": "Hello",
        },
    ]
    member_messages = [
        {
            "role": "system",
            "content": member_system_prompt_template.substitute(**profile),
        },
        {
            "role": "assistant",
            "content": "Hello",
        },
    ]
    # run interaction turns
    clinician_thinking_traces = []
    member_thinking_traces = []
    for turn in range(n_turns):
        # clinician turn
        clinician_response, clinician_thinking_trace = (
            clinician_model.generate_with_thinking(clinician_messages)
        )
        if not clinician_response:  # problematic for some apis
            clinician_response = "..."
        clinician_thinking_traces.append(clinician_thinking_trace)
        clinician_messages.append({"role": "assistant", "content": clinician_response})
        member_messages.append({"role": "user", "content": clinician_response})
        # member turn
        member_response, member_thinking_trace = member_model.generate_with_thinking(
            member_messages
        )
        if not member_response:  # problematic for some apis
            member_response = "..."
        member_thinking_traces.append(member_thinking_trace)
        member_messages.append({"role": "assistant", "content": member_response})
        clinician_messages.append({"role": "user", "content": member_response})
    return (
        clinician_messages[1:],
        clinician_thinking_traces,
        member_thinking_traces,
    )  # ignore system prompt


def main(
    profiles_path: str,
    clinician_system_template_version: str,
    clinician_model_api_params: dict[str, Any],
    member_system_template_version: str,
    member_model_api_params: dict[str, Any],
    n_turns: int,
    max_workers: int,
    output_path: str,
):
    # load data
    base_profiles = load_jsonl(profiles_path)
    profiles = []
    for p in base_profiles:
        in_dict = p["member_attributes"]
        in_dict["member_narrative"] = p["member_narrative"]
        profiles.append(in_dict)
    # instantiate system templates and models
    clinician_system_prompt_template = INTERACTION_CLINICIAN_VERSION_DICT[
        clinician_system_template_version
    ]
    clinician_model = InferenceEngine(clinician_model_api_params)
    member_system_prompt_template = INTERACTION_MEMBER_VERSION_DICT[
        member_system_template_version
    ]
    member_model = InferenceEngine(member_model_api_params)
    # run interactions in parallel
    all_interactions = []

    # Create a wrapper function to pass the profile index for ordering
    def process_profile(idx_profile):
        idx, profile = idx_profile
        interaction, clinician_thinking_traces, member_thinking_traces = (
            run_interactions(
                profile,
                clinician_system_prompt_template,
                clinician_model,
                member_system_prompt_template,
                member_model,
                n_turns,
            )
        )
        return idx, {
            "member_profile": profile,
            "interaction": interaction,
            "clinician_thinking_traces": clinician_thinking_traces,
            "member_thinking_traces": member_thinking_traces,
        }

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their indices
        futures = {
            executor.submit(process_profile, (idx, profile)): idx
            for idx, profile in enumerate(profiles)
        }

        # Create a dictionary to store results in order
        results = {}

        # Process completed futures with progress bar
        for future in tqdm(
            as_completed(futures), total=len(profiles), desc="Generating interactions"
        ):
            idx, result = future.result()
            results[idx] = result

    # Append results in original order
    for idx in sorted(results.keys()):
        all_interactions.append(results[idx])
    # save output
    save_to_jsonl(all_interactions, output_path)


if __name__ == "__main__":
    CLI([main], as_positional=False)
