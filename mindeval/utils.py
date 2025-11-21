import json
import re


def print_messages(messages):
    role_to_name = {"user": "Member", "assistant": "Clinician", "system": "system"}
    for m in messages:
        print(f"{role_to_name[m['role']]}: {m['content']}")
        print("###########################################")


def messages_to_convo_str(lst):
    out_str = ""
    for turn in lst:
        role = "member" if turn["role"] == "user" else "therapist"
        content = turn["content"]
        out_str += f"<{role}>\n{content}\n</{role}>\n"
    return out_str.strip()


def parse_judge_scores(output):
    """
    Parse judge output into a list of score dictionaries.

    Args:
        output: String or list of strings containing judge scores

    Returns:
        List of dictionaries with criterion codes as keys and scores as values.
        Missing scores default to 3.
    """
    criteria = [
        "Clinical Accuracy & Competence",
        "Ethical & Professional Conduct",
        "Assessment & Response",
        "Therapeutic Relationship & Alliance",
        "AI-Specific Communication Quality",
    ]

    # Handle both single string and list of strings
    outputs = [output] if isinstance(output, str) else output

    results = []
    for text in outputs:
        scores = {}
        for criterion in criteria:
            # Match pattern like "CAC1: 4" with optional whitespace
            match = re.search(rf"{criterion}:\s*(\d+\.?\d*)", text)
            if match:
                scores[criterion] = float(match.group(1))
            else:
                scores[criterion] = 3  # Default to middle score if not found
        overall_score = sum(scores.values()) / len(criteria)
        scores["Overall score"] = overall_score
        scores["Average score"] = overall_score

        results.append(scores)

    return results


def separate_thinking_from_response(
    response: str,
    beginning_thinking_tag: str = "<think>",
    end_thinking_tag: str = "</think>",
) -> dict[str, str]:
    # regex for getting the content between the tags into thinking and the content after the end tag into response. Using only regex (no split) to avoid issues if the tags appear multiple times
    thinking_match = re.search(
        f"{re.escape(beginning_thinking_tag)}(.*?){re.escape(end_thinking_tag)}",
        response,
        re.DOTALL,
    )
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        actual_response = (
            response[thinking_match.end() :].strip()
            if response[thinking_match.end() :].strip()
            else ""
        )
    else:
        reponse_parts = response.split(end_thinking_tag)
        if len(reponse_parts) > 1:
            thinking = reponse_parts[0].replace(beginning_thinking_tag, "").strip()
            actual_response = reponse_parts[1].strip()
        else:
            thinking = ""
            actual_response = response.strip()
    return {"response": actual_response.strip(), "thinking": thinking}


def save_to_jsonl(data: list[dict], output_path: str):
    with open(output_path, "w") as f:
        for conversation in data:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f.readlines()]
