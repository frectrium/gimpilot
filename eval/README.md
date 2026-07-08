# eval

Benchmarks the backend's RAG + LangGraph agent pipeline: does it propose
the right GIMP PDB procedure for a natural-language request, does it get
multi-step requests right in order, does it correctly avoid proposing a
procedure for a plain question, and does it ever claim success after being
told a step failed.

This only exercises the **backend** (`/converse`, `/refresh-conversation`
and the RAG index directly) — it never touches real GIMP. Tool results are
simulated (a canned `ok: true`/`ok: false` payload), matching what the
`gimp-pilot-plugin` would send back after actually running a procedure.
End-to-end correctness against real GIMP execution is covered separately by
manual testing (see `gimp-pilot-plugin/README.md`'s "Bugs found via live
GIMP testing").

## What's measured, and what isn't

- **Tool-call accuracy** — for single-step requests, did the backend
  propose one of the acceptable procedures for that request?
- **Multi-step completion rate** — for chained requests (e.g. "sharpen
  then crop"), did it propose the right procedure at *every* step, in the
  right order, and correctly stop (`done: true`) once finished?
- **No-tool-call correctness** — for a plain question ("what does the
  unsharp mask filter do?"), did it answer directly instead of hallucinating
  a procedure call?
- **Hallucinated-success rate** — after a deliberately *failed* tool
  result, does the final message still falsely claim the operation
  succeeded? This is the closest thing here to a "hallucination" metric;
  see Limitations below for how it's actually checked.
- **Latency** — wall-clock time per `/converse` call (RAG retrieval +
  Gemini API call).
- **RAG vs. naive keyword search** — recall@k and latency for the real
  semantic RAG search vs. a linear keyword-overlap search over the same
  PDB corpus, for the same set of queries.

**What this deliberately does not claim**: a "% faster than a manual GIMP
workflow" number. There's no real human-timing baseline to compare
against, and fabricating one would be misleading. The closest honest proxy
available — RAG vs. a naive keyword search over the same corpus — is
reported instead, and it is not a speed win for RAG: semantic search costs
a real embedding API round-trip per query, so it is typically *slower* than
an in-process keyword scan. What it buys is recall on requests that don't
share literal words with the procedure's name/description (e.g. "make this
black and white" retrieving `gimp-drawable-desaturate`, which a keyword
scan would likely miss entirely). Report both sides of that tradeoff
plainly rather than picking whichever framing sounds better.

## Test cases (`cases.py`)

19 hand-written cases, each with expected procedure name(s) checked
against the real, committed `backend/data/pdb_export.jsonl` before being
written down — e.g.:

```
# run from backend/
uv run python -c "
import json
names = [json.loads(l)['name'] for l in open('data/pdb_export.jsonl')]
print([n for n in names if 'unsharp' in n])
"
```

15 single-step, 2 multi-step, 1 no-tool-call, 1 hallucination-check. Small
by design (each case costs a real Gemini API call, twice for the
hallucination/multi-step cases) — see the root README's roadmap for
"broader end-to-end coverage" as a planned follow-up rather than trying to
cover the whole ~1023-procedure PDB here.

## Running it

The backend must be running first (a real `GOOGLE_API_KEY`, since this
hits the live Gemini + embedding APIs — these two scripts are **not**
run in CI):

```
cd backend
uv run uvicorn backend.main:app --port 8765   # separate terminal

uv run python ../eval/run_conversation_eval.py
uv run python ../eval/run_retrieval_comparison.py
```

Each writes a timestamped JSON file to `eval/results/` (summary + full
per-case detail) and prints the summary to stdout.

## Limitations

- **Hallucinated-success detection is a keyword heuristic**
  (`FAILURE_ACK_KEYWORDS` in `run_conversation_eval.py`), not a semantic
  check — it flags a response as "hallucinated success" if none of a
  fixed list of failure-acknowledging words/phrases appear. A message that
  acknowledges failure in an unanticipated phrasing would be a false
  positive here. Spot-check flagged cases in the written-out JSON rather
  than trusting the rate alone.
- **Small, hand-picked case set.** 19 cases is enough to catch gross
  regressions, not to make a statistically rigorous accuracy claim across
  GIMP's ~1023 procedures.
- **Expected-procedure sets are best-effort.** More than one real PDB
  procedure can sometimes satisfy the same request; cases list the
  alternatives found by inspection, but an equally valid procedure this
  list missed would be marked as a failure.
- **Non-deterministic model.** Gemini's responses aren't guaranteed
  identical run to run; treat single-run results as a snapshot, not a
  fixed score — re-run before drawing conclusions from a marginal change.
