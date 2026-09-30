"""llm.call_messages's params allow-list, checked against real litellm's own transformation (not a
stub, which cannot show whether a key is one litellm already maps for a given model) with
litellm.completion itself captured instead of called, so nothing goes over the network."""

from types import SimpleNamespace

import litellm

from mindeval import llm


def _capture(monkeypatch):
    calls = []

    def fake_completion(**kw):
        calls.append(kw)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               model=kw["model"], id="r")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(llm, "_LIB", litellm)
    return calls


def test_reasoning_effort_is_not_doubled_for_anthropic(monkeypatch):
    """litellm already translates reasoning_effort into Anthropic's `thinking`; forcing it through
    allowed_openai_params too made litellm send both, and Anthropic's real transform rejects the raw
    top-level field with a 400."""
    calls = _capture(monkeypatch)
    llm.call_messages([{"role": "user", "content": "hi"}], "anthropic/claude-sonnet-4-5", None, 1.0,
                      params={"reasoning_effort": "high"})
    kw = calls[0]
    assert "reasoning_effort" not in (kw.get("allowed_openai_params") or [])
    built = litellm.utils.get_optional_params(model="claude-sonnet-4-5", custom_llm_provider="anthropic",
                                              reasoning_effort=kw.get("reasoning_effort"),
                                              allowed_openai_params=kw.get("allowed_openai_params"))
    assert "reasoning_effort" not in built and built.get("thinking")  # the real Anthropic request body


def test_a_key_litellm_does_not_map_for_the_model_is_still_forced_through(monkeypatch):
    """cache_control_injection_points (v1's own judge config, and config/patient.yaml's claude entry)
    is not one of litellm's translated openai-style params for Anthropic, so drop_params would remove
    it silently without allowed_openai_params naming it."""
    calls = _capture(monkeypatch)
    points = [{"location": "message", "role": "system"}]
    llm.call_messages([{"role": "user", "content": "hi"}], "anthropic/claude-sonnet-4-5", None, 1.0,
                      params={"cache_control_injection_points": points})
    kw = calls[0]
    assert kw.get("allowed_openai_params") == ["cache_control_injection_points"]
    assert kw.get("cache_control_injection_points") == points


def test_vllm_extra_body_still_passes(monkeypatch):
    calls = _capture(monkeypatch)
    llm.call_messages([{"role": "user", "content": "hi"}], "hosted_vllm/patient", "http://vllm.invalid/v1", 1.0,
                      params={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}})
    kw = calls[0]
    assert kw.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "extra_body" not in (kw.get("allowed_openai_params") or [])
