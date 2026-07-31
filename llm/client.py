"""LLM provider interface: one call surface, two backends, per-task model routing.

Backends:
  cli  -- any Claude Code-compatible CLI (``claude -p --output-format json``);
          binary from $CLAUDE_CLI_BIN (default ``claude``).
  api  -- official ``anthropic`` SDK; needs $ANTHROPIC_API_KEY.

Model routing: every AI task in this repo is named in ``config.yaml`` under
``llm.tasks.<task>.model`` and can be overridden per run with
``LLM_MODEL_<TASK>`` (uppercased) or a CLI ``--model`` flag on the entrypoints.

All responses are cached on disk keyed by (model, prompt); eval-set caches are
committed so the repo reproduces offline with no key, no CLI, and no cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "llm" / "eval" / "cache"

# API list prices (USD per MTok, 2026-07) used to report *API-equivalent* cost.
# CLI-backend runs may bill differently (or be subscription-covered); the
# harness labels cost accordingly.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def load_config() -> dict[str, Any]:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def resolve_model(task: str, override: str | None = None) -> str:
    """Precedence: explicit --model flag > LLM_MODEL_<TASK> env > config.yaml."""
    if override:
        return override
    env = os.environ.get(f"LLM_MODEL_{task.upper()}")
    if env:
        return env
    cfg = load_config()["llm"]["tasks"]
    if task not in cfg:
        raise KeyError(f"unknown LLM task {task!r}; add it to config.yaml llm.tasks")
    return cfg[task]["model"]


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    duration_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None  # API-equivalent when tokens known, else None
    cached: bool = False

    def parsed_json(self) -> Any:
        """Parse the response as JSON, tolerating a fenced code block."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            t = t.rsplit("```", 1)[0]
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object in response: {t[:200]!r}")
        # strict=False tolerates literal control characters inside strings
        # (models occasionally emit raw newlines in memo_markdown)
        return json.loads(t[start : end + 1], strict=False)


def _api_equivalent_cost(model: str, tin: int | None, tout: int | None) -> float | None:
    if tin is None or tout is None or model not in PRICES:
        return None
    pin, pout = PRICES[model]
    return round((tin * pin + tout * pout) / 1e6, 6)


class ClaudeCLIClient:
    """Subprocess to a Claude Code-compatible CLI in print mode."""

    backend = "cli"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("CLAUDE_CLI_BIN", "claude")

    def complete(self, prompt: str, model: str, timeout_s: int = 420) -> LLMResponse:
        t0 = time.monotonic()
        proc = subprocess.run(
            [self.binary, "-p", "--output-format", "stream-json", "--verbose",
             "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        dur = int((time.monotonic() - t0) * 1000)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {proc.stderr[:500]}")
        result: dict[str, Any] | None = None
        tin: int | None = None
        tout = 0
        saw_usage = False
        for line in proc.stdout.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "assistant":
                u = d.get("message", {}).get("usage") or {}
                if "output_tokens" in u:
                    saw_usage = True
                    tout = max(tout, int(u["output_tokens"]))
                    cand = int(u.get("input_tokens", 0)) + int(
                        u.get("cache_read_input_tokens", 0) or 0
                    ) + int(u.get("cache_creation_input_tokens", 0) or 0)
                    tin = max(tin or 0, cand)
            elif d.get("type") == "result":
                result = d
        if result is None:
            raise RuntimeError(f"no result event from {self.binary}: {proc.stdout[-300:]}")
        if result.get("is_error"):
            raise RuntimeError(f"CLI error result: {result.get('result', '')[:500]}")
        if not saw_usage:
            tin, tout = None, None  # type: ignore[assignment]
        return LLMResponse(
            text=result["result"],
            model=model,
            backend=self.backend,
            duration_ms=result.get("duration_ms", dur),
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=_api_equivalent_cost(model, tin, tout),
        )


class AnthropicClient:
    """Official SDK backend."""

    backend = "api"

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, model: str, timeout_s: int = 300) -> LLMResponse:
        t0 = time.monotonic()
        msg = self._client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        dur = int((time.monotonic() - t0) * 1000)
        tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
        return LLMResponse(
            text="".join(b.text for b in msg.content if b.type == "text"),
            model=model,
            backend=self.backend,
            duration_ms=dur,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=_api_equivalent_cost(model, tin, tout),
        )


def get_client(backend: str | None = None):
    backend = backend or os.environ.get("LLM_BACKEND") or load_config()["llm"]["backend"]
    if backend == "cli":
        return ClaudeCLIClient()
    if backend == "api":
        return AnthropicClient()
    raise ValueError(f"unknown backend {backend!r}")


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()[:32]


def complete_cached(
    prompt: str,
    *,
    task: str,
    model: str | None = None,
    client=None,
    cache_dir: Path = CACHE_DIR,
    offline: bool = False,
    max_retries: int = 2,
) -> LLMResponse:
    """Cache-first completion. ``offline=True`` raises instead of calling out
    (CI uses this: the committed cache must fully cover the eval set)."""
    model = resolve_model(task, model)
    key = cache_key(model, prompt)
    path = cache_dir / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return LLMResponse(**{**d["response"], "cached": True})
    if offline:
        raise FileNotFoundError(f"offline mode: no cached response for {task}/{model}/{key}")
    client = client or get_client()
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.complete(prompt, model=model)
            break
        except Exception as e:  # noqa: BLE001 - retry then surface
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2**attempt)
    else:  # pragma: no cover
        raise last_err  # type: ignore[misc]
    cache_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task": task,
        "model": model,
        "backend": resp.backend,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response": {
            "text": resp.text,
            "model": resp.model,
            "backend": resp.backend,
            "duration_ms": resp.duration_ms,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
        },
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True))
    return resp
