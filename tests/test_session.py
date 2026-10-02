"""The counselor's own earlier turns go back as Anthropic accepts them: a turn with one thinking block exactly as
returned, a turn with several (Claude interleaving thinking and text) without its thinking."""

import asyncio
from types import SimpleNamespace

import pytest

from mindeval import engine, session


def _history_sent(monkeypatch, transcript):
    sent = []

    async def retry_text(models, role, messages, spec, what):
        sent.append(messages)
        return "ok", {}

    monkeypatch.setattr(engine, "retry_text", retry_text)
    counselor = session.Counselor(spec=SimpleNamespace(model="vertex_ai/claude-opus-5"), template=None)
    asyncio.run(counselor.respond({}, "system prompt", transcript))
    return sent[0]


def _block(signature):
    return {"type": "thinking", "thinking": "", "signature": signature}


def test_only_a_multi_block_turn_goes_back_without_its_thinking(monkeypatch):
    """Found on Claude Opus 5: litellm merges an interleaved turn's text and lists its thinking blocks, then
    rebuilds it thinking-first, which Anthropic rejects as a modified turn. Every later counselor call of the
    session then failed with a 400."""
    # The transcript is the member's: its "user" turns are the counselor's own.
    transcript = [{"role": "assistant", "content": "hi"},
                  {"role": "user", "content": "one block", "thinking_blocks": [_block("a")]},
                  {"role": "assistant", "content": "ok"},
                  {"role": "user", "content": "two blocks", "thinking_blocks": [_block("b"), _block("c")]},
                  {"role": "assistant", "content": "and?"}]
    sent = _history_sent(monkeypatch, transcript)
    counselor_turns = [m for m in sent if m["role"] == "assistant"]
    assert counselor_turns[0]["thinking_blocks"] == [_block("a")]
    assert "thinking_blocks" not in counselor_turns[1]
    assert counselor_turns[1]["content"] == "two blocks"


def test_a_single_block_turn_reaches_anthropic_in_its_original_order(monkeypatch):
    """A one-block turn is carried because litellm rebuilds it exactly as Claude returned it: thinking, then text.
    Checked against litellm's real Anthropic transform, not a stub."""
    anthropic_chat = pytest.importorskip("litellm.llms.anthropic.chat.transformation")
    transcript = [{"role": "assistant", "content": "hi"},
                  {"role": "user", "content": "one block", "thinking_blocks": [_block("a")]},
                  {"role": "assistant", "content": "and?"}]
    sent = _history_sent(monkeypatch, transcript)
    out = anthropic_chat.AnthropicConfig().transform_request(model="claude-opus-5", messages=sent,
                                                             optional_params={}, litellm_params={}, headers={})
    turn = [m for m in out["messages"] if m["role"] == "assistant"][0]
    assert [block["type"] for block in turn["content"]] == ["thinking", "text"]
    assert turn["content"][0]["signature"] == "a"
