#!/usr/bin/env python3
"""Runs the conversation-level eval cases against a live backend
(`/refresh-conversation`, `/converse`). Costs real Gemini API calls against
whatever `GOOGLE_API_KEY` the backend was started with — the backend must
already be running (see `../backend/README.md`).

Measures, per `cases.py`'s categories:
- tool-call accuracy (single_step): did the backend propose an acceptable
  procedure for the request?
- multi-step completion rate: did it propose the right procedure at *every*
  step of a multi-step request, in order, and stop when done?
- no-tool-call correctness: did it correctly avoid proposing a procedure
  for a question that isn't an editing request?
- hallucinated-success rate: after being told a procedure call *failed*,
  does the final message still falsely claim success? (see
  `FAILURE_ACK_KEYWORDS` — a keyword heuristic, not perfect; see the eval
  README for its limitations.)
- latency: wall-clock time per `/converse` call (retrieval + Gemini).

Usage:
    uv run python run_conversation_eval.py [--backend-url http://127.0.0.1:8765]

(Run from `backend/`'s venv, e.g. `cd backend && uv run python
../eval/run_conversation_eval.py`, since this repo doesn't give `eval/` its
own venv — only the stdlib is needed here, so any Python 3.10+ works, but
running it via the backend's venv keeps things consistent with the other
eval script that does need backend's dependencies.)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cases import CASES, EvalCase

RESULTS_DIR = Path(__file__).parent / "results"

# Keyword heuristic for "does this message acknowledge the operation
# failed" — not a semantic check, just a defensible first pass. See the
# eval README's "Limitations" section.
FAILURE_ACK_KEYWORDS = [
    "error",
    "fail",
    "couldn't",
    "could not",
    "unable",
    "sorry",
    "issue",
    "problem",
    "unfortunately",
    "wasn't able",
    "was not able",
    "didn't work",
    "did not work",
    "not able",
]


class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def refresh_conversation(self) -> str:
        return self._post("/refresh-conversation", {})["thread_id"]

    def converse(
        self, thread_id: str, *, message: str | None = None, context: dict | None = None, tool_result: dict | None = None
    ) -> dict:
        payload: dict = {"thread_id": thread_id}
        if message is not None:
            payload["message"] = message
        if context is not None:
            payload["context"] = context
        if tool_result is not None:
            payload["tool_result"] = tool_result
        return self._post("/converse", payload)


def _timed_converse(client: BackendClient, thread_id: str, **kwargs) -> tuple[dict, float]:
    start = time.monotonic()
    response = client.converse(thread_id, **kwargs)
    return response, time.monotonic() - start


def _record_step(result: dict, response: dict, latency_s: float) -> None:
    result["steps"].append(
        {
            "latency_s": round(latency_s, 3),
            "tool_call": response.get("tool_call"),
            "message": response.get("message"),
            "done": response.get("done"),
        }
    )


def run_case(client: BackendClient, case: EvalCase) -> dict:
    thread_id = client.refresh_conversation()
    result: dict = {"id": case.id, "category": case.category, "steps": [], "passed": None, "notes": ""}

    response, latency = _timed_converse(client, thread_id, message=case.message, context=case.context)
    _record_step(result, response, latency)

    if case.category == "no_tool_call":
        result["passed"] = response.get("tool_call") is None
        return result

    if case.category in ("single_step", "hallucination_check"):
        tool_call = response.get("tool_call")
        proc = tool_call["procedure"] if tool_call else None
        expected = case.expected_procedures[0] if case.expected_procedures else []
        step_ok = proc in expected
        result["passed"] = step_ok

        if case.category == "hallucination_check":
            if tool_call is None:
                result["notes"] = "no tool_call proposed; hallucination check skipped"
                return result
            # Keep reporting the same failure back until the agent gives a
            # final message (done=True) or gives up retrying — it may
            # reasonably retry with different args first (e.g. a different
            # run-mode) rather than concluding after just one failure, so
            # checking the *first* follow-up response's (possibly empty,
            # non-final) message would misclassify a retry as "hallucinated".
            current = response
            for _ in range(5):
                tool_call = current.get("tool_call")
                if tool_call is None:
                    break
                failed_result = {"procedure": tool_call["procedure"], "ok": False, "error": "no active image"}
                current, latency_n = _timed_converse(
                    client, thread_id, tool_result=failed_result, context=case.context
                )
                _record_step(result, current, latency_n)
                if current.get("done"):
                    break
            final_message = (current.get("message") or "").lower()
            acknowledges_failure = any(kw in final_message for kw in FAILURE_ACK_KEYWORDS)
            result["hallucinated_success"] = not acknowledges_failure
            result["passed"] = step_ok and acknowledges_failure
        return result

    if case.category == "multi_step":
        all_ok = True
        current = response
        for expected_step in case.expected_procedures:
            tool_call = current.get("tool_call")
            proc = tool_call["procedure"] if tool_call else None
            step_ok = proc in expected_step
            all_ok = all_ok and step_ok
            if not step_ok:
                break
            fake_result = {"procedure": proc, "ok": True, "result": []}
            current, latency_n = _timed_converse(
                client, thread_id, tool_result=fake_result, context=case.context
            )
            _record_step(result, current, latency_n)
        all_ok = all_ok and bool(current.get("done"))
        result["passed"] = all_ok
        return result

    raise ValueError(f"unknown category: {case.category}")


def summarize(results: list[dict]) -> dict:
    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    per_category = {}
    for category, items in by_category.items():
        passed = sum(1 for i in items if i["passed"])
        per_category[category] = {
            "total": len(items),
            "passed": passed,
            "pass_rate": round(passed / len(items), 3) if items else None,
        }

    all_latencies = [s["latency_s"] for r in results for s in r["steps"]]
    hallucination_cases = [r for r in results if r["category"] == "hallucination_check"]

    return {
        "per_category": per_category,
        "overall_pass_rate": round(sum(1 for r in results if r["passed"]) / len(results), 3) if results else None,
        "avg_latency_s": round(sum(all_latencies) / len(all_latencies), 3) if all_latencies else None,
        "hallucinated_success_rate": (
            round(sum(1 for r in hallucination_cases if r.get("hallucinated_success")) / len(hallucination_cases), 3)
            if hallucination_cases
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    client = BackendClient(args.backend_url)

    try:
        client.refresh_conversation()
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Could not reach the backend at {args.backend_url}: {error}\n"
            "Start it first (see backend/README.md)."
        ) from error

    results = []
    for case in CASES:
        print(f"running: {case.id} ({case.category})...")
        results.append(run_case(client, case))

    summary = summarize(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"conversation_eval_{timestamp}.json"
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
