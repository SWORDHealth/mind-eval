import litellm

from mindeval.utils import separate_thinking_from_response


class InferenceEngine:
    def __init__(self, api_params: dict[str, str]):
        self.api_params = api_params  # model, api_key, api_base

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = litellm.completion(
            messages=messages, max_retries=100, **self.api_params
        )
        return response.choices[0].message.content

    def batch_generate(self, list_of_messages: list[list[dict[str, str]]]) -> list[str]:
        responses = litellm.batch_completion(
            messages=list_of_messages, max_retries=100, **self.api_params
        )
        return [response.choices[0].message.content for response in responses]

    def generate_with_thinking(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        response = None
        while response is None:
            response = self.generate(messages)
            if response is None:
                print("Retrying generation because response is None...")
        parts = separate_thinking_from_response(response)
        return parts["response"], parts["thinking"]

    def batch_generate_with_thinking(
        self, list_of_messages: list[list[dict[str, str]]]
    ) -> tuple[list[str], list[str]]:
        responses = self.batch_generate(list_of_messages)
        texts = []
        thinking_traces = []
        for response in responses:
            parts = separate_thinking_from_response(response)
            texts.append(parts["response"])
            thinking_traces.append(parts["thinking"])
        return texts, thinking_traces
