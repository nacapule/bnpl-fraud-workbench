"""Command-line model backends for the memo drafter, run in isolation.

The drafter calls a model through a command-line tool the author is signed in to:

* ``codex``: the Codex CLI (``codex exec``), binary from ``CODEX_CLI_BIN``
  (default ``codex``), credentials from ``CODEX_AUTH_FILE`` (default
  ``$CODEX_HOME/auth.json``, else ``~/.codex/auth.json``);
* ``claude``: a Claude Code-compatible CLI in print mode, binary from
  ``CLAUDE_CLI_BIN`` (default ``claude``).

Both are general agents. A memo call must see the system prompt, the packet and
nothing else, and must not be able to read a file, run a command or use any
other tool. Every call therefore runs isolated:

Codex
  a temporary ``CODEX_HOME`` holding only a link to the credentials file (no
  instructions file, configuration, memories, skills or plugins) and a
  temporary ``HOME``; a minimal environment; an empty temporary working
  directory; ``--ephemeral --ignore-user-config --ignore-rules``, a read-only
  sandbox; every feature the CLI reports enabled switched off (``codex
  features list``); web search, user-input requests and skill instructions
  off; the model's catalog entry copied from the CLI's bundled catalog with
  every tool interface removed (no code mode, sub-agents, extra tools, patch,
  search or shell tool); the system prompt in place of the CLI's own
  instructions (``model_instructions_file``); the reasoning effort set
  explicitly.
Claude
  ``--setting-sources ""`` (no user, project or local settings, so none of
  their hooks, plugins, permissions or environment), ``--tools ""`` (no
  built-in tools), ``--strict-mcp-config`` without a configuration (no MCP
  servers), ``--disable-slash-commands`` (no skills), ``--system-prompt`` in
  place of the CLI's own prompt, ``--effort`` explicit, CLAUDE.md files and
  auto-memory off, an empty temporary working directory and a minimal
  environment (the process basics, the sign-in variables if set, and any
  names listed in ``CLAUDE_CLI_ENV`` for a launcher that needs them). Hook
  events are requested in the log and count as protocol failures. The CLI
  still adds a short environment note and the signed-in account's email
  address to the context, so callers scan every response for private terms
  (:func:`private_matches`).

Neither backend constrains decoding to a schema: the response format is part
of the system prompt and is validated afterwards, the same way for both.

Each call returns the response text, the full event log (kept outside the
repository by the caller) and an :class:`EventSummary` (CLI version, model,
effort, tools offered, tool, file, hook and error events, tokens) that is stored
with the response. A response whose log shows any tool, file, hook or error event,
or an event the summary does not recognise, is a protocol failure
(:attr:`EventSummary.protocol_ok`); the caller quarantines it.
:meth:`CodexBackend.fingerprint` and :meth:`ClaudeBackend.fingerprint` give the
CLI version and the hash of the isolation settings without a model call, so a
caller can pin both before it looks anything up.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BACKENDS = ("codex", "claude")
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Event types in a Codex event log (``codex exec --json``); anything else is not
# recognised and fails the call.
CODEX_EVENTS = frozenset({"thread.started", "turn.started", "turn.completed", "turn.failed",
                          "item.started", "item.updated", "item.completed", "error"})
CODEX_FAILURE_EVENTS = frozenset({"turn.failed", "error"})
# Item types that are the model's own output, not tool use; every other item type
# (web search, plans, unknown ones) counts as a tool event.
CODEX_MESSAGE_ITEMS = frozenset({"agent_message", "reasoning"})
CODEX_FILE_ITEMS = frozenset({"command_execution", "file_change", "mcp_tool_call"})

# The model's catalog entry, with every tool interface removed.
CODEX_MODEL_OVERRIDES: Mapping[str, Any] = {
    "tool_mode": "direct",  # no code-mode exec/wait
    "multi_agent_version": None,  # no sub-agent tools or role instructions
    "experimental_supported_tools": [],  # no clock, no asynchronous messages
    "apply_patch_tool_type": None,
    "supports_search_tool": False,
    "shell_type": "disabled",
}
CODEX_CONFIG = (
    'web_search="disabled"',
    "tools.experimental_request_user_input={enabled=false}",
    "skills.include_instructions=false",
)
CODEX_FLAGS = (
    "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
    "--sandbox", "read-only", "--json",
)
CLAUDE_FLAGS = ("-p", "--output-format", "stream-json", "--verbose", "--include-hook-events",
                "--setting-sources", "", "--tools", "", "--strict-mcp-config",
                "--disable-slash-commands")
CLAUDE_SETTINGS_ENV = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
                       "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
# Event types in a Claude stream-json log; anything else is not recognised.
CLAUDE_EVENTS = frozenset({"system", "assistant", "user", "result", "ping", "stream_event",
                           "rate_limit_event"})
CLAUDE_TEXT_BLOCKS = frozenset({"text", "thinking", "redacted_thinking"})
CLAUDE_TOOL_BLOCKS = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})
# Environment variables a CLI may need for the network, passed through when set.
NETWORK_ENV = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy",
               "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR")
# What a process needs to run at all, passed through when set.
PROCESS_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "LC_ALL",
               "LC_CTYPE", "TERM")
# Where Claude Code finds its sign-in when it is not in the default place.
CLAUDE_AUTH_ENV = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR")


class BackendError(RuntimeError):
    """The CLI failed (transport, exit status, or no answer); never a scored outcome.

    ``called`` is False when the failure came before the model was asked (the CLI's
    own setup), so the call is not charged to a budget."""

    def __init__(self, message: str, *, called: bool = True):
        super().__init__(message)
        self.called = called


@dataclass(frozen=True)
class Request:
    backend: str
    model: str
    effort: str
    system_prompt: str
    prompt: str
    timeout_s: int = 900

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, not {self.effort!r}")


@dataclass(frozen=True)
class EventSummary:
    """What a call's event log shows; stored with every response."""

    backend: str
    cli_version: str
    model: str
    effort: str
    n_events: int
    event_counts: Mapping[str, int]
    tools_offered: int | None  # None when the log does not list them (Codex)
    n_tool_events: int
    n_file_events: int
    n_error_events: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    isolation: str = ""  # sha256 of the isolation settings used
    n_hook_events: int = 0
    n_unrecognized_events: int = 0

    @property
    def protocol_ok(self) -> bool:
        return (self.n_tool_events == 0 and self.n_file_events == 0
                and self.n_error_events == 0 and self.n_hook_events == 0
                and self.n_unrecognized_events == 0 and not self.tools_offered)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_counts": dict(sorted(self.event_counts.items())),
                "protocol_ok": self.protocol_ok}


@dataclass(frozen=True)
class Response:
    text: str
    summary: EventSummary
    duration_ms: int
    events: tuple[Mapping[str, Any], ...] = field(repr=False, default=())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _run(command: Sequence[str], *, env: Mapping[str, str] | None = None,
         cwd: Path | None = None, stdin: str | None = None,
         timeout_s: int = 120, called: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a command; ``called`` says whether it asks the model (for the error)."""
    try:
        return subprocess.run(list(command), input=stdin, capture_output=True, text=True,
                              env=None if env is None else dict(env), cwd=cwd,
                              timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as error:
        raise BackendError(f"{command[0]} timed out after {timeout_s}s",
                           called=called) from error
    except OSError as error:
        raise BackendError(f"cannot run {command[0]}: {error}", called=False) from error


def _jsonl(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def private_matches(text: str, terms: Iterable[str]) -> list[str]:
    """The private terms (an author's name or address) that occur in ``text``."""
    lowered = text.lower()
    return sorted({term for term in terms if term and term.lower() in lowered})


def _first_line(result: subprocess.CompletedProcess[str]) -> str:
    lines = (result.stdout.strip() or result.stderr.strip()).splitlines()
    return lines[0].strip() if lines else ""


# ----------------------------------------------------------------------------- Codex


def parse_feature_list(listing: str) -> list[tuple[str, str, bool]]:
    """``codex features list`` rows as (name, stage, enabled)."""
    rows = []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[-1] not in ("true", "false"):
            continue
        rows.append((parts[0], " ".join(parts[1:-1]), parts[-1] == "true"))
    return rows


def features_to_disable(listing: str) -> list[str]:
    """Every enabled feature that can still be switched (removed ones cannot)."""
    return sorted(name for name, stage, enabled in parse_feature_list(listing)
                  if enabled and stage != "removed")


def isolated_catalog(catalog: Mapping[str, Any], model: str) -> dict[str, Any]:
    """The catalog with ``model``'s entry stripped of every tool interface."""
    models = [dict(entry) for entry in catalog.get("models", [])]
    matches = [entry for entry in models if entry.get("slug") == model]
    if len(matches) != 1:
        raise BackendError(f"model {model!r} is not in the CLI's bundled catalog",
                           called=False)
    matches[0].update(CODEX_MODEL_OVERRIDES)
    return {**catalog, "models": models}


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def summarize_codex(events: Sequence[Mapping[str, Any]], *, cli_version: str, model: str,
                    effort: str, isolation: str) -> EventSummary:
    counts: Counter[str] = Counter()
    item_types: dict[str, str] = {}  # item id -> type, from its start or completion
    usage: Mapping[str, Any] = {}
    failures = unrecognized = 0
    for index, event in enumerate(events):
        kind = str(event.get("type"))
        item = event.get("item")
        if kind not in CODEX_EVENTS:
            unrecognized += 1
        elif kind in CODEX_FAILURE_EVENTS:
            failures += 1
        if kind.startswith("item."):
            if not isinstance(item, Mapping):
                unrecognized += 1
                counts[kind] += 1
                continue
            item_type = str(item.get("type"))
            counts[f"{kind}:{item_type}"] += 1
            item_types[str(item.get("id", f"#{index}"))] = item_type
        else:
            counts[kind] += 1
        if kind == "turn.completed" and isinstance(event.get("usage"), Mapping):
            usage = event["usage"]
    types = list(item_types.values())
    return EventSummary(
        backend="codex", cli_version=cli_version, model=model, effort=effort,
        n_events=len(events), event_counts=dict(counts), tools_offered=None,
        n_tool_events=sum(t not in CODEX_MESSAGE_ITEMS | CODEX_FILE_ITEMS | {"error"}
                          for t in types),
        n_file_events=sum(t in CODEX_FILE_ITEMS for t in types),
        n_error_events=types.count("error") + failures,
        input_tokens=_token_count(usage.get("input_tokens")),
        output_tokens=_token_count(usage.get("output_tokens")),
        isolation=isolation, n_unrecognized_events=unrecognized,
    )


def codex_final_text(events: Sequence[Mapping[str, Any]]) -> str | None:
    texts = [event["item"].get("text") for event in events
             if event.get("type") == "item.completed"
             and isinstance(event.get("item"), Mapping)
             and event["item"].get("type") == "agent_message"]
    return texts[-1] if texts else None


class CodexBackend:
    name = "codex"

    def __init__(self, binary: str | None = None, auth_file: str | Path | None = None):
        self.binary = binary or os.environ.get("CODEX_CLI_BIN", "codex")
        default_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        self.auth_file = Path(auth_file or os.environ.get("CODEX_AUTH_FILE")
                              or default_home / "auth.json")

    def _environment(self, home: Path, codex_home: Path, tmp: Path) -> dict[str, str]:
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home),
               "CODEX_HOME": str(codex_home), "TMPDIR": str(tmp)}
        env.update({key: os.environ[key] for key in NETWORK_ENV if key in os.environ})
        return env

    def version(self) -> str:
        return _first_line(_run([self.binary, "--version"], timeout_s=60))

    def fingerprint(self, model: str, effort: str) -> dict[str, str]:
        """The CLI version and the isolation hash a call with this model would have."""
        with tempfile.TemporaryDirectory(prefix="memo-") as name:
            *_, settings = self.prepare(Path(name), Request("codex", model, effort, "", ""))
        return {"cli_version": self.version(), "isolation": sha256_text(canonical_json(settings))}

    def prepare(self, base: Path, request: Request) -> tuple[list[str], dict[str, str], Path,
                                                               dict[str, Any]]:
        """Build the isolated command, environment and working directory in ``base``."""
        home, codex_home, work, tmp = (base / name for name in
                                       ("home", "codex-home", "work", "tmp"))
        for directory in (home, codex_home, work, tmp):
            directory.mkdir()
        if not self.auth_file.is_file():
            raise BackendError(f"no Codex credentials at {self.auth_file}", called=False)
        (codex_home / "auth.json").symlink_to(self.auth_file.resolve())
        env = self._environment(home, codex_home, tmp)

        listing = _run([self.binary, "features", "list"], env=env, cwd=work)
        if listing.returncode != 0:
            raise BackendError(f"codex features list failed: {listing.stderr[-300:]}",
                               called=False)
        disabled = features_to_disable(listing.stdout)
        bundled = _run([self.binary, "debug", "models", "--bundled"], env=env, cwd=work)
        if bundled.returncode != 0:
            raise BackendError(f"codex debug models failed: {bundled.stderr[-300:]}",
                               called=False)
        catalog = isolated_catalog(json.loads(bundled.stdout), request.model)
        (base / "catalog.json").write_text(json.dumps(catalog))
        (base / "instructions.md").write_text(request.system_prompt)

        command = [self.binary, "exec", *CODEX_FLAGS]
        for feature in disabled:
            command += ["--disable", feature]
        for setting in CODEX_CONFIG:
            command += ["-c", setting]
        command += ["-c", f"model_catalog_json={json.dumps(str(base / 'catalog.json'))}",
                    "-c", f"model_instructions_file={json.dumps(str(base / 'instructions.md'))}",
                    "-m", request.model, "-c", f"model_reasoning_effort={request.effort}"]
        command.append("-")
        settings = {
            "flags": list(CODEX_FLAGS), "config": list(CODEX_CONFIG),
            "disabled_features": disabled, "model_overrides": dict(CODEX_MODEL_OVERRIDES),
            "instructions": "system prompt", "environment": sorted(env),
        }
        return command, env, work, settings

    def complete(self, request: Request) -> Response:
        version = self.version()
        with tempfile.TemporaryDirectory(prefix="memo-") as name:
            base = Path(name)
            command, env, work, settings = self.prepare(base, request)
            started = time.monotonic()
            result = _run(command, env=env, cwd=work, stdin=request.prompt,
                          timeout_s=request.timeout_s, called=True)
            duration = int((time.monotonic() - started) * 1000)
        events = _jsonl(result.stdout)
        summary = summarize_codex(events, cli_version=version, model=request.model,
                                  effort=request.effort,
                                  isolation=sha256_text(canonical_json(settings)))
        text = codex_final_text(events)
        if result.returncode != 0 or text is None:
            raise BackendError(f"codex exited {result.returncode} with "
                               f"{'no answer' if text is None else 'an answer'}: "
                               f"{(result.stderr or '')[-300:]}")
        return Response(text=text, summary=summary, duration_ms=duration, events=tuple(events))


# ---------------------------------------------------------------------------- Claude


def summarize_claude(events: Sequence[Mapping[str, Any]], *, cli_version: str, model: str,
                     effort: str, isolation: str) -> EventSummary:
    counts: Counter[str] = Counter()
    tools_offered: int | None = None
    tool = file = error = hook = unrecognized = 0
    usage: Mapping[str, Any] = {}
    reported_model = model
    for event in events:
        kind = str(event.get("type"))
        subtype = event.get("subtype")
        counts[f"{kind}:{subtype}" if subtype else kind] += 1
        message = event.get("message") if isinstance(event.get("message"), Mapping) else {}
        content = message.get("content") or []
        blocks = [block for block in content if isinstance(block, Mapping)] \
            if isinstance(content, list) else []
        if kind not in CLAUDE_EVENTS:
            unrecognized += 1
        elif kind == "system" and str(subtype).startswith("hook"):
            hook += 1
        elif kind == "system" and subtype == "init":
            offered = list(event.get("tools") or []) + list(event.get("mcp_servers") or [])
            tools_offered = len(offered)
        elif kind == "assistant":
            if message.get("model"):
                reported_model = str(message["model"])
            tool += sum(block.get("type") in CLAUDE_TOOL_BLOCKS for block in blocks)
            unrecognized += sum(block.get("type") not in CLAUDE_TOOL_BLOCKS | CLAUDE_TEXT_BLOCKS
                                for block in blocks)
        elif kind == "user":
            file += sum(block.get("type") == "tool_result" for block in blocks)
        elif kind == "result":
            usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
            error += bool(event.get("is_error")) or subtype not in (None, "success")
    input_parts = [_token_count(usage.get(key)) for key in (
        "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
    return EventSummary(
        backend="claude", cli_version=cli_version, model=reported_model, effort=effort,
        n_events=len(events), event_counts=dict(counts), tools_offered=tools_offered,
        n_tool_events=tool, n_file_events=file, n_error_events=error,
        input_tokens=(sum(part or 0 for part in input_parts)
                      if input_parts[0] is not None else None),
        output_tokens=_token_count(usage.get("output_tokens")),
        isolation=isolation, n_hook_events=hook, n_unrecognized_events=unrecognized,
    )


def claude_final_text(events: Sequence[Mapping[str, Any]]) -> str | None:
    results = [event for event in events if event.get("type") == "result"]
    text = results[-1].get("result") if results else None
    return text if isinstance(text, str) and text else None


class ClaudeBackend:
    name = "claude"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("CLAUDE_CLI_BIN", "claude")

    def version(self) -> str:
        """The Claude Code version that answers. When ``CLAUDE_CLI_BIN`` is a launcher
        rather than Claude Code itself, the version is that of the ``claude`` on the
        path, which the launcher runs, plus a hash of the launcher's own version line."""
        own = _first_line(_run([self.binary, "--version"], timeout_s=60))
        if "Claude Code" in own:
            return own
        code = "Claude Code not on the path"
        if shutil.which("claude"):
            code = _first_line(_run(["claude", "--version"], timeout_s=60))
        return f"{code}; launcher {sha256_text(own)[:12]}"

    def fingerprint(self, model: str, effort: str) -> dict[str, str]:
        """The CLI version and the isolation hash a call with this model would have."""
        *_, settings = self.prepare(Request("claude", model, effort, "", ""))
        return {"cli_version": self.version(), "isolation": sha256_text(canonical_json(settings))}

    @staticmethod
    def _environment() -> tuple[dict[str, str], list[str]]:
        extra = [name.strip() for name in os.environ.get("CLAUDE_CLI_ENV", "").split(",")
                 if name.strip()]
        names = [*PROCESS_ENV, *NETWORK_ENV, *CLAUDE_AUTH_ENV, *extra]
        env = {name: os.environ[name] for name in names if name in os.environ}
        env.update(CLAUDE_SETTINGS_ENV)
        passed = sorted(name for name in env if name not in PROCESS_ENV)
        return env, passed

    def prepare(self, request: Request) -> tuple[list[str], dict[str, str], dict[str, Any]]:
        command = [self.binary, *CLAUDE_FLAGS, "--model", request.model,
                   "--effort", request.effort, "--system-prompt", request.system_prompt]
        env, passed = self._environment()
        settings = {"flags": list(CLAUDE_FLAGS), "system_prompt": "replaced",
                    "environment": dict(CLAUDE_SETTINGS_ENV), "process": list(PROCESS_ENV),
                    "passed": passed}
        return command, env, settings

    def complete(self, request: Request) -> Response:
        version = self.version()
        command, env, settings = self.prepare(request)
        with tempfile.TemporaryDirectory(prefix="memo-") as name:
            started = time.monotonic()
            result = _run(command, env=env, cwd=Path(name), stdin=request.prompt,
                          timeout_s=request.timeout_s, called=True)
            duration = int((time.monotonic() - started) * 1000)
        events = _jsonl(result.stdout)
        summary = summarize_claude(events, cli_version=version, model=request.model,
                                   effort=request.effort,
                                   isolation=sha256_text(canonical_json(settings)))
        text = claude_final_text(events)
        if result.returncode != 0 or text is None:
            raise BackendError(f"{self.binary} exited {result.returncode} with "
                               f"{'no answer' if text is None else 'an answer'}: "
                               f"{(result.stderr or '')[-300:]}")
        return Response(text=text, summary=summary, duration_ms=duration, events=tuple(events))


def backend(name: str) -> CodexBackend | ClaudeBackend:
    if name == "codex":
        return CodexBackend()
    if name == "claude":
        return ClaudeBackend()
    raise ValueError(f"unknown backend {name!r}")
