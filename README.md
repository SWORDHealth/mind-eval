# Mind-Eval
A benchmark for AI mental-health counselors, played against a simulated patient over multiple sessions.

<p align="center">
  <img src="assets/mindsim.png" width="50%" alt="MindSim, the patient-simulation harness: a patient LM, conditioned on a fixed profile and a per-session scenario, commits to a move with a tool call and then verbalises it to the counselor LM under test; a shared memory, updated after each session, is read by both in the next; the transcripts of every session are scored by the judge.">
</p>

## What's New
> **📢 September 2026 — MindEval 2.** `master` is now a multi-session benchmark with a new patient
> simulator and a new judge. **Scores are not comparable with MindEval 1**, which stays unchanged on
> the [`legacy`](https://github.com/SWORDHealth/mind-eval/tree/legacy) branch.

- **Multi-session** — 50 synthetic members, 3 sessions each, with a memory log carried between sessions.
- **Patient simulator** — each member picks its next move with a tool call before it writes its turn,
  played from a clinical formulation and a situation that evolves between sessions.
- **MQM judge** — every counselor reply gets an MQM severity from GPT-6.1 Sol (or a judge model of your
  choice), with the prompt that ships in the repo; the benchmark reports the mean and p99 conversation
  severity over every member × session (lower is better).
- **Both judges** — `--judge_version mindeval1` still scores each session with the original
  five-criterion rubric.
- **Pinned arcs** — each member's schedule, time context and per-call seeds are drawn from its id,
  `--seed` and a fixed namespace, never from the config's bytes: every counselor meets the same arcs
  (the ones MindEval 2 was developed on), and an edit such as a comment never re-draws the benchmark.

## Installation
```bash
git clone https://github.com/SWORDHealth/mind-eval.git
cd mind-eval
uv sync
source .venv/bin/activate
cp .env.example .env
```
>Python 3.12; [uv](https://docs.astral.sh/uv/).

>Fill in `.env` with the model, endpoint and key of the patient, the counselor and the judge. They
>are read from the environment only, and keys are never written to the outputs.

**IMPORTANT**: the counselor's system prompt is the text file `examples/counselor_system.j2`.
You can change anything in it, or pass your own with `--counselor_system_prompt_path <file>`, but
keep the memory field `{{ memory }}` exactly as it is, once. That is where the counselor receives its
log of the earlier sessions, so every counselor gets the same memory. In session 1 it expands to
"No earlier sessions: this is your first conversation with this member."; later, to "Your log of
previous sessions with this member, oldest first. Lines tagged `[open]` are things the member said
they would do and has not yet reported back on." followed by one line per entry, written after each
session. A prompt without the field, or with it more than once, is refused before the run starts.

## Run interactions
```bash
python mindeval/scripts/generate_interactions.py --output_dir runs/<counselor>
```
>Defaults: the 50 members in `data/profiles.jsonl`, `--sessions 3`, `--max_turns 30` (counselor
>replies; the member may leave sooner), `--max_workers 16`, `--seed 0`. See `--help`.

>The patient is DeepSeek-V4.1-Flash by default, served by vLLM (`hosted_vllm/`) with tool calling that
>uses a named `tool_choice` by default. For a served model that only accepts automatic tool choice,
>pass `--patient_force_move false`; use `true` to explicitly request a named `commit_move`. The
>preflight requires a real tool call in either mode, and the selected mode must match on resume.
>When a situation carries an agenda (most do), the member is also
>offered `complete_todo`/`add_todo` with `tool_choice: "auto"` between the move and the words, so a
>vLLM patient needs `--enable-auto-tool-choice` (and a matching `--tool-call-parser`) for that path
>too; the preflight only exercises the selected commit path. Another model can
>play it: set `MINDEVAL_PATIENT_MODEL`, e.g. `vertex_ai/claude-sonnet-4-6`. Add anything else a model
>needs with `--patient_params`. A `vertex_ai/` model needs `uv sync --extra vertex`.

>Provider-specific counselor settings go in `--counselor_params`, e.g. `'{"reasoning_effort": "high"}'`.

>Each counselor gets its own earlier replies back as its API returned them, reasoning included, as
>models trained on preserved thinking expect. OpenAI reasoning models return it only through the
>Responses API: run them as `azure/responses/<deployment>` (or `openai/responses/<model>`) with
>`--counselor_params '{"include": ["reasoning.encrypted_content"], "store": false}'`.

>Rerun the same command to resume: finished members are kept, and unfinished sessions restart.

>For a style experiment, pass `--patient_style_path <file.yaml>` with a complete alternative to
>`mindeval/config/style2.yaml`. It may add `move_descriptions: {do_answer_question: "..."}` for existing
>moves; archetype exclusions still apply. The selected file's contents change the config fingerprint,
>and its resolved path and SHA-256 are recorded in `run.json` and session injections. Changing the
>selection or its contents requires a fresh output directory; the seed namespace and arcs stay pinned.
>Omitting the flag preserves the benchmark's default style and fingerprint.

The optional candidates in `experiments/patient_styles/` are `nemotron_context.yaml`,
`nemotron_agency.yaml`, and `nemotron_private_notes.yaml`. Context removes universal
under-answering; agency adds a cue for the patient's practical concern and relevant corrections;
private notes keeps the context policy and treats the assigned story as background rather than
an ordered script. The context candidate was selected for validation on unused synthetic profiles
after a matched development comparison; the private-note cue did not show a supported advantage.
Select it with `--patient_style_path experiments/patient_styles/nemotron_context.yaml`. This does
not establish parity with the historical Sonnet/DeepSeek table.

Patient speech containing a standalone `end_conversation` YAML/JSON field is rejected and retried
within the native budget. Rejected text and requested retry seeds remain in the trace. If valid
speech cannot be recovered, the session fails without inserting a fabricated patient reply.
Inline mentions, quoted literals and fenced examples remain allowed.

## Audit patient realism (experimental)

The patient audit in `experiments/realism/` reviews delivered patient turns using causal prefixes
(turns 1, 4, 8 and the last), then complete sessions, then recurring patterns across sessions.
It hides model labels and private profiles from the judge, verifies exact quoted evidence, and
reports failed audits and coverage. MindSim JSONL needs one terminal, non-error session footer
for full-session and cross-session eligibility; live/failed prefixes receive causal turn audits only
and are reported as incomplete. Duplicate or out-of-order turns fail loading. Generic delivered
JSON/JSONL is complete by input convention, with termination unverified. Short answers, guarded
disclosure, fluent language and quick relief are not automatic failures. This is a new, unvalidated patient rubric, separate from the
counselor MQM score; using Dawn's judge endpoint does not validate the new rubric. Development
audits showed ceiling scores, missed failures and source allegations that did not survive manual
checks. Review the cited preceding counselor turn and full assigned source before accepting a finding;
keep uncertain allegations and invalid judge results visible.

Create a manifest in your ignored run directory; paths resolve relative to the manifest:
```json
{"groups": {"candidate": [{"path": "members/example/arc/ep000/trace.jsonl", "case_id": "example"}]}}
```
Run from the repository root, first with `--dry-run` to verify inputs without API calls:
```bash
python -m experiments.realism.hierarchical_realism --manifest runs/<experiment>/patient_manifest.json \
  --output-dir runs/<experiment>/patient_audit --api-base http://<judge-host>:<port>/v1 \
  --judge-model dawn-judge --dry-run
```
Remove `--dry-run` to judge. The runner sends one user message at temperature 0 with a 16,384-token
output limit, retains invalid attempts, and caches by input bytes, rendered prompt and judge settings.
Supply endpoint credentials through `MINDEVAL_JUDGE_API_KEY` (or `--api-key-env`). Keep human-reference
transcripts and judge outputs in ignored local run directories; use development references while
iterating, and freeze the candidate before evaluating a held-out pool.

A separate `experiments.realism.patient_grounding` runner compares completed MindSim JSONL traces
with assigned YAML sources. Add `profile_path` and `situation_path` to each manifest item and pass the
same command flags. Unsourced human controls are not applicable. It distinguishes contradiction,
source recitation and committed-move mismatch from ordinary added detail or a concern that was never
surfaced; those observations do not lower the source-free realism score.

The isolated continuity pilot in `experiments/continuity/` replays the first completed sessions
of a paired development smoke run. It records exact prompts, source hashes, native call settings,
seeds, raw attempts and failure coverage for the original and candidate memory prompts:
```bash
python -m experiments.continuity.replay_compaction --smoke runs/<paired-smoke> \
  --out runs/<memory-replay> --synthetic
```
This prepares inputs without model calls; add `--execute` to run. The current input layout is
`{dawn_agency,glm53_agency}/members/{m000004,m000013}/arc/ep000/` plus each arm's `run.json` and
`arc.json`. Three optional hand-written ambiguity/commitment probes are counted separately from
these four completed conversations. This compaction runner does not execute evolution or alter
benchmark defaults.

A separate paired evolution replay uses the same four smoke histories to generate session 2
situations. Preparation proves the original request matches `evolve_situation`, freezes the runtime
and inputs, and makes no model calls:

```bash
python -m experiments.continuity.replay_evolution --smoke runs/<paired-smoke> \
  --out runs/<evolution-replay>
python runs/<evolution-replay>/snapshot/runtime/experiments/continuity/replay_evolution.py \
  --smoke runs/<paired-smoke> --out runs/<evolution-replay> --execute
```

Execution uses one serial worker and retains native phase seeds, call policy, raw attempts, parsed
situations and actual HTTP payload audits. Review newly generated events against the full inputs;
ordinary details and small plan changes are allowed. These four previously reviewed cases do not
establish broader realism or prompt efficacy, particularly when the original does not reproduce a
historical error. The default evolution prompt remains unchanged.

The opt-in [complete spoken history package](experiments/continuity/full_history/README.md)
retains every earlier patient/counselor message with session and speaker labels, without a model
summary call. It provides hard-fail history budgets and an optional source-evidence calendar solver
for a frozen three-session research pipeline. The original eight-arc, 24-session comparison was
reviewed on two familiar synthetic profiles. The current optional package adds chronology
metadata and source/call binding guards, verified offline; behavior on fresh generated dialogues
remains untested. Availability does not promote it to the benchmark default.


The separate [fresh-campaign runner](experiments/continuity/fresh_campaign.py) prepares and
verifies a registered eight-profile, four-arm, three-session comparison (32 arcs/96 sessions).
Preparation binds the exact selection, source/calendar review, candidate code, native call policies,
templates and installed runtime. Verification uses local files with networking denied before SDK
imports. Future execution requires a separately reviewed authorization bound to the exact campaign;
each attempted case preserves complete or failed output evidence and refuses implicit replay.

```bash
python -m experiments.continuity.fresh_campaign verify --root <frozen-campaign>
```

## Run judgments
```bash
python mindeval/scripts/generate_judgments.py --output_dir runs/<counselor>
```
>The judge is GPT-6.1 Sol by default: `.env.example` sets `MINDEVAL_JUDGE_MODEL=openai/gpt-6.1-sol` (it
>needs `OPENAI_API_KEY`; on Azure, use `azure/<deployment>`), and the script's defaults are its settings:
>`--judge_params '{"reasoning_effort": "high"}'`, no temperature, `--judge_max_tokens 32768`. Another
>judge model goes in `MINDEVAL_JUDGE_*`, but its scores aren't comparable with the benchmark's.

>The judge prompt is in `mindeval/judge_prompts.py`. A system message holds the task, severities, rubric
>and answer format, identical on every call so providers can cache it (`MINDEVAL2_JUDGE_SYSTEM_PROMPT`).
>A user message holds only the example: the conversation so far and the reply to judge
>(`MINDEVAL2_JUDGE_USER_PROMPT`). `MINDEVAL_JUDGE_PROMPT` replaces both with one template in a file, sent
>as a single user message; verdicts from any other prompt aren't comparable with the benchmark's.
>Rerun to resume. It can run while `generate_interactions.py` is still going: only finished sessions
>are judged. The first pass preflights: a judge that can't return a usable verdict for any of the
>first three units stops before spending the rest of the budget. A unit that keeps failing never
>blocks the ones after it; it stays pending.

>Provider-specific judge settings go in `--judge_params` (`'{}'` sends none); like
>`--counselor_params`/`--patient_params` it can't set `model`, `api_base` or `api_key`. It's part of
>the judge's identity, like the model, prompt, temperature and max tokens: once a verdict exists, a
>later pass with different params is refused (exit 2). Move `judge/<version>/` aside to judge again
>with other settings.

The original five-criterion rubric takes a judge model the same way. Its `<member_details>` is the counselor's own memory for that session — what it
received in `{{ memory }}`, not the member's private profile — so a counselor that correctly recalls
session 1 isn't scored as though it hallucinated:
```bash
python mindeval/scripts/generate_judgments.py --output_dir runs/<counselor> --judge_version mindeval1
```

## Check scores
```bash
python -c "import json; print(json.load(open('runs/<counselor>/judge/mindeval2/summary.json'))['mqm'])"
```
The score is the **MQM conversation severity**, and lower is better. Each reply is weighted by
severity: No concern 0, Minor 1, Major 5, Critical 10, averaged within each session and then over
every member × session, so one long session can't outweigh several short ones. The averaging also
dilutes a single severe reply in a long session (one Critical among 30 clean replies scores 0.33), so
`mqm.incidents` reports beside it the share of sessions with at least one Critical reply (`critical`)
and with at least one Major or Critical (`major_or_worse`). `summary.json` also has severity and issue
rates.

```bash
python -c "import json; print(json.load(open('runs/<counselor>/judge/mindeval1/summary.json'))['means'])"
```
The v1 rubric scores each session on five criteria plus their mean (`Overall score`), all 1-6 where
higher is better; `summary.json` gives the mean of each over every member × session.

## Do it all in one go
```bash
bash run_benchmark.sh runs/<counselor> [mindeval2|mindeval1]
```
>Runs interactions then judgments for the given judge version (default `mindeval2`), then prints the
>summary. A refused interactions pass (exit 2) stops before judging anything; a partial one (exit 1)
>is still judged, and the script's own exit code reports the worse of the two stages.

## Outputs
```
runs/<counselor>/
  run.json                  what ran: models, endpoints, prompt and profile hashes, settings
  members/<id>/arc/ep00N/   per session: trace.jsonl (every turn), situation.yaml, log.md (the memory after it)
  judge/mindeval2/          MQM: judge.json, verdicts.jsonl, summary.json
  judge/mindeval1/          five-criterion rubric: judge.json, verdicts.jsonl, summary.json
```

## Data and license
The 50 members are synthetic. Their persona seeds come from
[nvidia/Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) (CC BY
4.0). A language model turned them into clinical formulations, Claude Sonnet 4.6 wrote the session-1
situations, and a handful of words were edited by hand.

The code is licensed under [Apache 2.0](LICENSE). Models and data are licensed under CC BY-NC-SA 4.0,
including the member profiles and the MQM judge prompt (`MINDEVAL2_JUDGE_SYSTEM_PROMPT` and
`MINDEVAL2_JUDGE_USER_PROMPT`). MindEval 1's profiles, human annotations and meta-evaluation data are
unchanged on the [`legacy`](https://github.com/SWORDHealth/mind-eval/tree/legacy) branch.

## Cite
MindEval 2:
```bibtex
@misc{mendonca2026mindeval2,
      title={MindEval2: Benchmarking Language Models on Multi-Session Mental Health Support},
      author={John Mendonça and Daniel Tschernutter and José Pombal and Areti Vassilopoulos and Catarina Botelho and Maya D'Eon and Nuno Guerreiro and Ricardo Rei},
      year={2026},
      note={Preprint},
      url={https://github.com/SWORDHealth/mind-eval},
}
```

MindEval 1:
```bibtex
@misc{pombal2025mindevalbenchmarkinglanguagemodels,
      title={MindEval: Benchmarking Language Models on Multi-turn Mental Health Support}, 
      author={José Pombal and Maya D'Eon and Nuno M. Guerreiro and Pedro Henrique Martins and António Farinhas and Ricardo Rei},
      year={2025},
      eprint={2511.18491},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.18491}, 
}
```
