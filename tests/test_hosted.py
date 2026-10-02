"""A mindeval session hosted elsewhere: UserSim's hosted runtime (ProbeEpisodeRuntime) pauses at every model call
and an outside host answers it. Checked with UserSim's own parity test, which runs one session standalone and once
hosted and compares the rows and every prompt."""

import asyncio
import json
from pathlib import Path

import pytest
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import EpisodeContractError
from usersim.testing import ScriptedModel, assert_hosted_parity

from mindeval.arc import counselor_memory
from mindeval.config import CallSpec
from mindeval.models import MemberRow
from mindeval.probe import Runner, SessionInput, trajectory_id
from mindeval.session import Counselor

ROOT = Path(__file__).resolve().parents[1]
ROW = MemberRow.model_validate_json((ROOT / "data/profiles.jsonl").read_text().splitlines()[0])
MOVE = {"reasoning": "answer it", "move": "do_answer_question"}


def _spec(model):
    return CallSpec(model=model, api_base=None, temperature=1.0, max_tokens=None, timeout=60, max_retries=2)


def _session() -> tuple[dict, dict]:
    """The persona and the rest of a session's row, with no trace path: a hosted session writes only its row."""
    counselor = Counselor.from_text(_spec("host/counselor"), (ROOT / "examples/counselor_system.j2").read_text())
    patient = _spec("host/patient")
    inp = SessionInput.of(counselor=counselor, patient=patient, member=ROW, situation=ROW.situation,
                          knobs=ROW.knobs, seed=7, max_turns=2, counselor_memory=counselor_memory([]))
    row = Runner(patient, counselor.spec, max_turns=2).row(inp, trajectory=trajectory_id(ROW.member_id, "a", 0, 0),
                                                           episode=0)
    return json.loads(row.pop("persona")), row


class ContentPatient(ScriptedModel):
    """A host's patient that answers the move as JSON content, as a server that ignores tool_choice does."""

    async def acompletion(self, messages, **kwargs):
        reply = await super().acompletion(messages, **kwargs)
        if self.role == "user" and any(t["function"]["name"] == "commit_move" for t in kwargs.get("tools") or []):
            reply.message.content = json.dumps(MOVE)
        return reply


class NativePatient(ScriptedModel):
    """A host's patient that answers the move as a native tool call, as mindeval's own patients do."""

    async def acompletion(self, messages, **kwargs):
        reply = await super().acompletion(messages, **kwargs)
        if self.role == "user" and any(t["function"]["name"] == "commit_move" for t in kwargs.get("tools") or []):
            reply.message.content = ""
            reply.message.tool_calls = [{"id": f"c{len(self.prompts)}", "type": "function",
                                         "function": {"name": "commit_move", "arguments": json.dumps(MOVE)}}]
        return reply


CONFIG = ConversationSimulatorConfig(name="conversation", max_turns=2, context_compression=False)


def test_a_session_runs_the_same_when_hosted():
    persona, data = _session()
    asyncio.run(assert_hosted_parity("mindeval", config=CONFIG, persona=persona, data=data,
                                     model_factory=ContentPatient))


@pytest.mark.xfail(strict=True, raises=EpisodeContractError,
                   reason="UserSim's hosted runtime records tool calls only from the assistant (episode_runtime.py, "
                          "_validate_recorded_response), and mindeval's patient commits its move as one")
def test_a_patient_committing_by_tool_call_can_be_hosted():
    persona, data = _session()
    asyncio.run(assert_hosted_parity("mindeval", config=CONFIG, persona=persona, data=data,
                                     model_factory=NativePatient))
