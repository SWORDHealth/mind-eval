# Mind-Eval
A benchmark for AI mental-health counselors, played against a simulated patient over multiple sessions.

## What's New
> **📢 September 2026 — MindEval 2.** `master` is now a multi-session benchmark with a new patient
> simulator and a new judge. **Scores are not comparable with MindEval 1**, which stays unchanged on
> the [`legacy`](https://github.com/SWORDHealth/mind-eval/tree/legacy) branch.

- **Multi-session** — 50 synthetic members, 3 sessions each, with a memory log carried between sessions.
- **Patient simulator** — each member picks its next move with a tool call before it writes its turn,
  played from a clinical formulation and a situation that evolves between sessions.
- **MindJudge-27B** — every counselor reply gets an MQM severity; the benchmark reports the mean and
  p99 conversation severity over every member × session (lower is better).
- **Both judges** — `--judge_version mindeval1` still scores each session with the original
  five-criterion rubric.
- **Same arcs as the internal build** — member schedules, time context and per-call seeds are pinned
  to match MindEval2 @ a18de8be exactly, so the two are still reproducible against each other;
  behaviour differs only by bug fixes made during this port's review (a judge preflight that could
  stall a run for good, a stray reply dropped mid-session, an avoided move still named in a member's
  own prompt, and a handful more — see the commit history on `mindeval2-port`).

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

**IMPORTANT**: the counselor's system prompt is the Jinja template `examples/counselor_system.j2`.
You can change anything in it, or pass your own with `--counselor_system_prompt_path <file.j2>`, but
keep the memory field `{{ memory }}` exactly as it is. That is where the counselor receives its log of
the earlier sessions (in session 1 it says there are none); the comment at the top of the file shows
what it expands to. To keep the comparison fair, a template may only be text around one `{{ memory }}`:
no other variables, filters or `{% %}` tags. Anything else is refused before the run starts.

## Run interactions
```bash
python mindeval/scripts/generate_interactions.py --output_dir runs/<counselor>
```
>Defaults: the 50 members in `data/profiles.jsonl`, `--sessions 3`, `--max_turns 30` (counselor
>replies; the member may leave sooner), `--max_workers 16`, `--seed 0`. See `--help`.

>The patient is DeepSeek-V4.1-Flash by default, served by vLLM (`hosted_vllm/`) with tool calling that
>honours a named `tool_choice`. When a situation carries an agenda (most do), the member is also
>offered `complete_todo`/`add_todo` with `tool_choice: "auto"` between the move and the words, so a
>vLLM patient needs `--enable-auto-tool-choice` (and a matching `--tool-call-parser`) for that path
>too, not just a named `tool_choice`; the preflight only exercises the named path. Another model can
>play it: set `MINDEVAL_PATIENT_MODEL`, e.g. `vertex_ai/claude-sonnet-4-6`. Add anything else a model
>needs with `--patient_params`. A `vertex_ai/` model needs `uv sync --extra vertex`.

>Provider-specific counselor settings go in `--counselor_params`, e.g. `'{"reasoning_effort": "high"}'`.

>Each counselor gets its own earlier replies back as its API returned them, reasoning included, as
>models trained on preserved thinking expect. OpenAI reasoning models return it only through the
>Responses API: run them as `azure/responses/<deployment>` (or `openai/responses/<model>`) with
>`--counselor_params '{"include": ["reasoning.encrypted_content"], "store": false}'`.

>Rerun the same command to resume: finished members are kept, and unfinished sessions restart.

## Run judgments
Serve MindJudge-27B and download its prompt (the download needs `HF_TOKEN` while the model is private):
```bash
vllm serve swordhealth/MindJudge-27B --language-model-only --served-model-name mindjudge \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 131072 --gpu-memory-utilization 0.9
uvx --from huggingface_hub hf download swordhealth/MindJudge-27B mindjudge_prompt.txt --local-dir mindjudge
```
Then:
```bash
python mindeval/scripts/generate_judgments.py --output_dir runs/<counselor>
```
>Rerun to resume. It can run while `generate_interactions.py` is still going: only finished sessions
>are judged. The first pass preflights: a judge that can't return one usable verdict stops before
>spending the rest of the budget. After that, a unit that keeps failing no longer blocks the ones
>after it.

>Provider-specific judge settings go in `--judge_params`, e.g. `'{"reasoning_effort": "high"}'`; like
>`--counselor_params`/`--patient_params` it can't set `model`, `api_base` or `api_key`, and it's part
>of the judge's identity, so a later pass with different params starts a new judge directory instead
>of mixing into the same verdicts.

The original five-criterion rubric needs no prompt download, just a judge model of your choice in
`MINDEVAL_JUDGE_*`. Its `<member_details>` is the counselor's own memory for that session — what it
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
every member × session, so one long session can't outweigh several short ones. `summary.json` also
has severity and issue rates.

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
[nvidia/Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)
(CC BY 4.0). A language model turned them into clinical formulations, Claude Sonnet 4.6 wrote the
session-1 situations, and a handful of words were edited by hand. The code is MIT; the profiles are
CC BY 4.0; MindJudge-27B and its prompt are CC BY-NC-SA 4.0. MindEval 1's profiles, human annotations
and meta-evaluation data are unchanged on the [`legacy`](https://github.com/SWORDHealth/mind-eval/tree/legacy)
branch.

## Cite
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
