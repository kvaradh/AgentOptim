"""First thing to run once an API key exists. Nothing here has ever executed.

`core.llm.complete_json` catches every exception and returns None, so an LLM
that is rate-limited, timing out, or emitting unparseable JSON looks exactly
like one that is switched off: the loop quietly falls back to deterministic
verdicts and still produces a demo. That is the right behaviour on stage and
the wrong behaviour while building, because you cannot tune prompts you cannot
see failing.

This probe reports what actually happened on every call:

    python -m scripts.probe_llm              # connectivity + one real round
    python -m scripts.probe_llm --rounds 3   # a short run, with call stats
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from core import llm


class Probe:
    """Wraps complete_json and records why each call succeeded or failed."""

    def __init__(self):
        self.events: list[dict] = []
        self._original = llm.complete_json

    def install(self):
        def wrapped(system: str, user: str, model: str | None = None,
                    max_tokens: int = llm.DEFAULT_MAX_TOKENS):
            started = time.time()
            record = {
                "input_chars": len(system) + len(user),
                "model": model or llm.DEFAULT_MODEL,
            }
            try:
                result = self._call_verbose(system, user, model, max_tokens)
                record.update(result)
            except Exception as exc:  # noqa: BLE001 - the whole point is to see it
                record.update(outcome="exception", error=f"{type(exc).__name__}: {exc}")
                result = {"payload": None}
            record["seconds"] = round(time.time() - started, 2)
            self.events.append(record)
            return record.get("payload")

        llm.complete_json = wrapped
        # Rebind the copies other modules imported by name at import time.
        import core.agents
        import core.edits
        import core.loop

        core.agents.complete_json = wrapped
        for module in (core.edits, core.loop):
            if hasattr(module, "complete_json"):
                module.complete_json = wrapped
        return self

    def _call_verbose(self, system, user, model, max_tokens) -> dict:
        """The body of complete_json, but reporting instead of swallowing."""
        if os.environ.get("AGENT_LLM", "").strip().lower() == "off":
            return {"outcome": "disabled", "payload": None}
        if not llm.available():
            reason = (
                "no ANTHROPIC_API_KEY"
                if not os.environ.get("ANTHROPIC_API_KEY")
                else "anthropic SDK not importable"
            )
            return {"outcome": "unavailable", "error": reason, "payload": None}

        response = llm._client().messages.create(
            model=model or llm.DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        stats = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "stop_reason": getattr(response, "stop_reason", None),
        }

        try:
            return {"outcome": "ok", "payload": json.loads(text), **stats}
        except json.JSONDecodeError:
            pass
        match = llm._JSON_BLOCK.search(text)
        if not match:
            return {
                "outcome": "no_json",
                "error": f"no JSON object in {len(text)} chars",
                "sample": text[:200],
                "payload": None,
                **stats,
            }
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {"outcome": "bad_json", "error": str(exc),
                    "sample": match.group(0)[:200], "payload": None, **stats}
        if not isinstance(parsed, dict):
            return {"outcome": "not_a_dict", "payload": None, **stats}
        return {"outcome": "ok_after_extraction", "payload": parsed, **stats}

    def report(self) -> int:
        if not self.events:
            print("No LLM calls were attempted.")
            return 1

        outcomes = Counter(event["outcome"] for event in self.events)
        total = len(self.events)
        print(f"\n{'=' * 68}\n{total} LLM calls\n")
        for outcome, count in outcomes.most_common():
            print(f"  {outcome:20s} {count:3d}  ({count / total:.0%})")

        failures = [e for e in self.events if e["outcome"] not in
                    ("ok", "ok_after_extraction")]
        if failures:
            print(f"\n{len(failures)} call(s) fell back to the deterministic path:")
            for event in failures[:6]:
                stop = event.get("stop_reason")
                print(
                    f"  [{event['outcome']}] {event.get('error', '')}"
                    + (f"  stop_reason={stop}" if stop else "")
                    + (f"  out_tokens={event['output_tokens']}"
                       if event.get("output_tokens") else "")
                )
                if event.get("sample"):
                    print(f"      model said: {event['sample']!r}")

        billed = [e for e in self.events if e.get("input_tokens")]
        if billed:
            tin = sum(e["input_tokens"] for e in billed)
            tout = sum(e["output_tokens"] or 0 for e in billed)
            # Sonnet 5 list price; adjust if AGENT_MODEL is set to something else.
            cost = tin / 1e6 * 2.0 + tout / 1e6 * 10.0
            print(
                f"\ntokens: {tin:,} in / {tout:,} out over {len(billed)} calls"
                f"   ~${cost:.4f} at Sonnet 5 rates"
            )
            slow = max(self.events, key=lambda e: e.get("seconds", 0))
            print(f"slowest call: {slow['seconds']}s")

        ok = outcomes["ok"] + outcomes["ok_after_extraction"]
        print(f"\n{'=' * 68}")
        if ok == 0:
            print("VERDICT: the LLM path is not working. Every call fell back.")
            return 1
        if ok < total:
            print(
                f"VERDICT: partially working -- {ok}/{total} succeeded. The rest "
                f"silently degraded. Fix those before tuning any prompt."
            )
            return 1
        print(f"VERDICT: the LLM path works. All {total} calls returned usable JSON.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the untested LLM path")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    print(f"model     : {llm.DEFAULT_MODEL}")
    print(f"api key   : {'present' if os.environ.get('ANTHROPIC_API_KEY') else 'MISSING'}")
    print(f"AGENT_LLM : {os.environ.get('AGENT_LLM') or '(unset -- LLM enabled)'}")
    print(f"timeout   : {llm.DEFAULT_TIMEOUT}s")

    if not llm.available():
        print(
            "\nThe LLM path cannot run. Set ANTHROPIC_API_KEY in the environment's "
            "settings (a new session picks it up), then run this again."
        )
        return 1

    probe = Probe().install()
    from core.loop import run

    print(f"\nRunning {args.rounds} round(s) with LLM agents...\n")
    record = run(rounds=args.rounds, verbose=True)

    status = probe.report()
    print(f"chose: {record['final']['smiles']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
