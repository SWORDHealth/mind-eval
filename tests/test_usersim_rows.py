"""What UserSim sees of a run: one row per session, stored as `usersim simulate` stores a run, read back the way
`usersim evaluate` reads one; and the probe found through its entry point."""

import contextlib
import io
import json
import tempfile
from importlib.metadata import entry_points
from pathlib import Path

import pandas as pd
from parity import _dies_then_resumes
from usersim.engine.core.probes import known_probes
from usersim.engine.core.storage import is_partitioned_directory, materialize_to_temp_file, resolve_run, run_subroot


def test_every_session_is_a_usersim_row_and_the_trace_is_its_conversation():
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            assert _dies_then_resumes(Path(tmp))["codes"] == [1, 0]
        out = Path(tmp) / "run"
        root = out / "usersim"
        run_id = resolve_run(root, None)  # as `usersim evaluate --trajectories <root>` resolves it
        assert run_id is not None and is_partitioned_directory(run_subroot(root, run_id))
        rows = pd.read_parquet(materialize_to_temp_file(run_subroot(root, run_id))).sort_values("episode")

        # each session once: session 2 as it ran again; the attempt that died is set aside, a trajectory of its own
        assert list(zip(rows.episode, rows.conversation_status)) == [(0, True), (1, True)]
        assert set(rows.probe_family) == {"mindeval"} and set(rows.locale) == {"en_US"}
        assert rows.persona_uuid.nunique() == 1
        (died,) = pd.read_parquet(root / "superseded" / f"run={run_id}").itertuples()
        assert died.episode == 1 and not died.conversation_status and died.trajectory_id not in set(rows.trajectory_id)
        outcome = json.loads(died.simulation_outcome)
        assert outcome["status"] == "failed" and outcome["failure_attribution"] == "user_model"
        assert outcome["failure_detail"].startswith("MemberError: speaking failed")

        arc = out / "members" / "m000119" / "arc"
        for (_, row), ep in zip(rows[rows.conversation_status].iterrows(), ("ep000", "ep001")):
            lines = [json.loads(line) for line in (arc / ep / "trace.jsonl").read_text().splitlines()]
            turns, session = lines[:-1], lines[-1]
            messages = json.loads(row.conversation_messages)
            assert messages[0] == {"role": "system", "content": session["counselor_system_prompt"]}
            spoken = [(m["role"], m["content"]) for m in messages[1:]]
            expected = [("user", turns[0]["utterance"])]
            for t in turns[1:]:
                expected += [("assistant", t["counselor_utterance"]), ("user", t["utterance"])]
            assert spoken == expected
            assert all(m.get("reasoning_content") for m in messages if m["role"] == "assistant")
            meta = json.loads(row.conversation_metadata)
            assert meta["session_id"] == session["session_id"] and meta["termination"] == session["termination"]
            # the row names its trace, but run again it writes none: replaying it cannot append to this run's trace
            assert row.mindeval_trace.endswith(f"{ep}/trace.jsonl")
            assert json.loads(row.mindeval_session)["trace_path"] is None


def test_usersim_finds_the_probe_through_its_entry_point():
    (point,) = [e for e in entry_points(group="usersim.probes") if e.name == "mindeval"]
    assert point.value == "mindeval.probe" and "mindeval" in known_probes()
