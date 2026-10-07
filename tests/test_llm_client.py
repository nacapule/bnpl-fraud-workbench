"""Live memo calls run isolated, and their event logs decide whether a response counts.

The command-line tools are general agents; these tests check the command, environment
and working directory each call gets, without running a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm import client

FEATURES = """\
apps                                     stable             true
code_mode                                under development  false
collaboration_modes                      removed            true
memories                                 stable             false
multi_agent                              stable             true
shell_tool                               stable             true
some_new_tool                            under development  true
"""
CATALOG = {"models": [
    {"slug": "gpt-6.1-sol", "tool_mode": "code_mode_only", "multi_agent_version": "v2",
     "experimental_supported_tools": ["clock"], "apply_patch_tool_type": "freeform",
     "supports_search_tool": True, "shell_type": "unified_exec", "base_instructions": "x"},
    {"slug": "other", "tool_mode": "code_mode_only"},
]}


def fake_run(calls):
    def run(command, *, env=None, cwd=None, stdin=None, timeout_s=120, called=False):
        calls.append(SimpleNamespace(command=list(command), env=env, cwd=cwd, stdin=stdin))
        if command[1:3] == ["features", "list"]:
            return SimpleNamespace(returncode=0, stdout=FEATURES, stderr="")
        if command[1:3] == ["debug", "models"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(CATALOG), stderr="")
        if command[1] == "--version":
            return SimpleNamespace(returncode=0, stdout="codex-cli 9.9.9\n", stderr="")
        events = [{"type": "thread.started"}, {"type": "turn.started"},
                  {"type": "item.completed", "item": {"id": "i0", "type": "agent_message",
                                                      "text": '{"ok": true}'}},
                  {"type": "turn.completed", "usage": {"input_tokens": 500,
                                                       "output_tokens": 90}}]
        assert cwd is not None and list(Path(cwd).iterdir()) == []  # empty working directory
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)),
                               stderr="")
    return run


@pytest.fixture
def auth(tmp_path: Path) -> Path:
    path = tmp_path / "auth.json"
    path.write_text("{}")
    return path


def test_codex_call_is_isolated(monkeypatch, auth: Path) -> None:
    calls: list = []
    monkeypatch.setattr(client, "_run", fake_run(calls))
    monkeypatch.setenv("SECRET_SETTING", "do-not-pass")
    request = client.Request("codex", "gpt-6.1-sol", "high", "SYSTEM TEXT", "PACKET")
    response = client.CodexBackend(binary="codex", auth_file=auth).complete(request)
    call = calls[-1]
    command = call.command
    for flag in ("--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                 "--ignore-rules", "--json"):
        assert flag in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("-m") + 1] == "gpt-6.1-sol"
    assert "model_reasoning_effort=high" in command
    disabled = [command[i + 1] for i, part in enumerate(command) if part == "--disable"]
    assert disabled == ["apps", "multi_agent", "shell_tool", "some_new_tool"]
    settings = [command[i + 1] for i, part in enumerate(command) if part == "-c"]
    for expected in client.CODEX_CONFIG:
        assert expected in settings
    assert any(s.startswith("model_catalog_json=") for s in settings)
    assert any(s.startswith("model_instructions_file=") for s in settings)
    assert call.stdin == "PACKET" and command[-1] == "-"
    assert "SYSTEM TEXT" not in command  # the system prompt goes in a file
    assert set(call.env) <= {"PATH", "HOME", "CODEX_HOME", "TMPDIR", *client.NETWORK_ENV}
    assert "SECRET_SETTING" not in call.env
    assert call.env["HOME"] != os.environ.get("HOME")
    assert Path(call.env["CODEX_HOME"]).name == "codex-home"
    assert response.text == '{"ok": true}' and response.summary.protocol_ok
    assert response.summary.cli_version == "codex-cli 9.9.9"
    assert response.summary.input_tokens == 500


def test_codex_home_holds_only_the_credentials_link(monkeypatch, auth: Path,
                                                    tmp_path: Path) -> None:
    monkeypatch.setattr(client, "_run", fake_run([]))
    base = tmp_path / "call"
    base.mkdir()
    request = client.Request("codex", "gpt-6.1-sol", "high", "SYSTEM", "PACKET")
    command, env, work, _ = client.CodexBackend(auth_file=auth).prepare(base, request)
    home = Path(env["CODEX_HOME"])
    assert [p.name for p in home.iterdir()] == ["auth.json"]
    assert (home / "auth.json").is_symlink()
    assert (home / "auth.json").resolve() == auth.resolve()
    assert (base / "instructions.md").read_text() == "SYSTEM"
    catalog = json.loads((base / "catalog.json").read_text())
    entry = next(m for m in catalog["models"] if m["slug"] == "gpt-6.1-sol")
    for key, value in client.CODEX_MODEL_OVERRIDES.items():
        assert entry[key] == value
    assert entry["base_instructions"] == "x"  # everything else unchanged
    other = next(m for m in catalog["models"] if m["slug"] == "other")
    assert other["tool_mode"] == "code_mode_only"


def test_unknown_model_or_missing_credentials_fail_before_any_call(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_run", fake_run([]))
    base = tmp_path / "a"
    base.mkdir()
    with pytest.raises(client.BackendError):
        client.CodexBackend(auth_file=tmp_path / "missing.json").prepare(
            base, client.Request("codex", "gpt-6.1-sol", "high", "S", "P"))
    (tmp_path / "auth.json").write_text("{}")
    base = tmp_path / "b"
    base.mkdir()
    with pytest.raises(client.BackendError):
        client.CodexBackend(auth_file=tmp_path / "auth.json").prepare(
            base, client.Request("codex", "no-such-model", "high", "S", "P"))


def test_feature_rows_with_two_word_stages_are_read() -> None:
    rows = client.parse_feature_list(FEATURES)
    assert ("code_mode", "under development", False) in rows
    assert client.features_to_disable(FEATURES) == ["apps", "multi_agent", "shell_tool",
                                                    "some_new_tool"]


def test_effort_is_always_explicit() -> None:
    with pytest.raises(ValueError):
        client.Request("codex", "gpt-6.1-sol", "", "S", "P")
    with pytest.raises(ValueError):
        client.Request("api", "claude-opus-5-5", "high", "S", "P")


def codex_summary(*items):
    events = [{"type": "thread.started"}]
    for index, item_type in enumerate(items):
        events.append({"type": "item.completed", "item": {"id": f"i{index}", "type": item_type}})
    return client.summarize_codex(events, cli_version="v", model="m", effort="high",
                                  isolation="i")


@pytest.mark.parametrize("items,ok", [
    (("reasoning", "agent_message"), True),
    (("command_execution", "agent_message"), False),
    (("file_change",), False),
    (("mcp_tool_call",), False),
    (("web_search", "agent_message"), False),
    (("error", "agent_message"), False),  # a refused tool call shows as an error item
])
def test_any_tool_file_or_error_event_is_a_protocol_failure(items, ok) -> None:
    assert codex_summary(*items).protocol_ok is ok


def test_a_started_tool_item_counts_once() -> None:
    events = [{"type": "item.started", "item": {"id": "t", "type": "command_execution"}},
              {"type": "item.completed", "item": {"id": "t", "type": "command_execution"}}]
    summary = client.summarize_codex(events, cli_version="v", model="m", effort="high",
                                     isolation="i")
    assert summary.n_file_events == 1


def claude_events(tools=(), blocks=("text",)):
    return [
        {"type": "system", "subtype": "init", "tools": list(tools), "mcp_servers": [],
         "model": "unknown", "claude_code_version": "unknown"},
        {"type": "assistant", "message": {"model": "claude-opus-5-5",
                                          "content": [{"type": b} for b in blocks]}},
        {"type": "result", "subtype": "success", "result": "{}", "is_error": False,
         "usage": {"input_tokens": 2, "cache_creation_input_tokens": 828,
                   "output_tokens": 300}},
    ]


def test_claude_summary_reads_tools_offered_and_the_answering_model() -> None:
    clean = client.summarize_claude(claude_events(), cli_version="2.1.0 (Claude Code)",
                                    model="m", effort="high", isolation="i")
    assert clean.protocol_ok and clean.tools_offered == 0
    assert clean.model == "claude-opus-5-5" and clean.cli_version == "2.1.0 (Claude Code)"
    assert clean.input_tokens == 830 and clean.output_tokens == 300
    offered = client.summarize_claude(claude_events(tools=("Read",)), cli_version="c",
                                      model="m", effort="high", isolation="i")
    assert not offered.protocol_ok
    used = client.summarize_claude(claude_events(blocks=("tool_use", "text")),
                                   cli_version="c", model="m", effort="high", isolation="i")
    assert not used.protocol_ok and used.n_tool_events == 1


def test_claude_call_has_no_tools_and_replaces_the_system_prompt(monkeypatch) -> None:
    backend = client.ClaudeBackend(binary="claude-cli")
    command, env, _ = backend.prepare(
        client.Request("claude", "claude-opus-5-5", "high", "SYSTEM", "PACKET"))
    assert command[command.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in command
    assert command[command.index("--system-prompt") + 1] == "SYSTEM"
    assert command[command.index("--effort") + 1] == "high"
    assert command[command.index("--model") + 1] == "claude-opus-5-5"
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert "PACKET" not in command  # the packet goes on standard input


def test_claude_call_loads_no_settings_hooks_plugins_or_skills(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_SETTING", "do-not-pass")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_SOMETHING", "1")
    monkeypatch.setenv("LAUNCHER_STATE", "kept")
    monkeypatch.setenv("CLAUDE_CLI_ENV", "LAUNCHER_STATE")
    command, env, settings = client.ClaudeBackend(binary="claude-cli").prepare(
        client.Request("claude", "claude-opus-5-5", "high", "SYSTEM", "PACKET"))
    assert command[command.index("--setting-sources") + 1] == ""  # no user/project/local
    assert "--disable-slash-commands" in command and "--include-hook-events" in command
    assert "--settings" not in command
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert "SECRET_SETTING" not in env and "CLAUDE_CODE_ENABLE_SOMETHING" not in env
    assert env["LAUNCHER_STATE"] == "kept" and "LAUNCHER_STATE" in settings["passed"]
    assert set(env) <= {*client.PROCESS_ENV, *client.NETWORK_ENV, *client.CLAUDE_AUTH_ENV,
                        *client.CLAUDE_SETTINGS_ENV, "LAUNCHER_STATE"}


@pytest.mark.parametrize("event", [
    {"type": "system", "subtype": "hook_started", "hook_name": "SessionStart"},
    {"type": "system", "subtype": "hook_response", "output": "context"},
])
def test_hook_events_fail_a_claude_call(event) -> None:
    events = claude_events()
    summary = client.summarize_claude([events[0], event, *events[1:]], cli_version="c",
                                      model="m", effort="high", isolation="i")
    assert summary.n_hook_events == 1 and not summary.protocol_ok


@pytest.mark.parametrize("events", [
    [{"type": "tool_progress"}],                                   # an unknown event type
    [{"type": "assistant", "message": {"content": [{"type": "image"}]}}],  # unknown block
    [{"type": "result", "subtype": "error_during_execution", "is_error": False}],
])
def test_unrecognised_or_failed_claude_events_fail_the_call(events) -> None:
    summary = client.summarize_claude([*claude_events(), *events], cli_version="c",
                                      model="m", effort="high", isolation="i")
    assert not summary.protocol_ok


@pytest.mark.parametrize("extra", [
    {"type": "error", "message": "stream disconnected"},
    {"type": "turn.failed", "error": {"message": "limit"}},
    {"type": "exec.started"},                                       # not a known event
    {"type": "item.completed"},                                     # an item without a body
])
def test_top_level_codex_failures_and_unknown_events_fail_the_call(extra) -> None:
    events = [{"type": "thread.started"}, {"type": "turn.started"}, extra,
              {"type": "item.completed", "item": {"id": "i0", "type": "agent_message",
                                                  "text": "{}"}},
              {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 5}}]
    summary = client.summarize_codex(events, cli_version="v", model="m", effort="high",
                                     isolation="i")
    assert not summary.protocol_ok
    assert summary.n_error_events + summary.n_unrecognized_events == 1


def test_claude_version_names_claude_code_and_hashes_a_launcher(monkeypatch) -> None:
    def run(command, **_kwargs):
        line = "2.1.9 (Claude Code)" if command[0] == "claude" else "some-launcher 0.3"
        return SimpleNamespace(returncode=0, stdout=line + "\n", stderr="")
    monkeypatch.setattr(client, "_run", run)
    monkeypatch.setattr(client.shutil, "which", lambda name: "/bin/" + name)
    assert client.ClaudeBackend(binary="claude").version() == "2.1.9 (Claude Code)"
    wrapped = client.ClaudeBackend(binary="some-launcher").version()
    assert wrapped.startswith("2.1.9 (Claude Code); launcher ")
    assert "some-launcher" not in wrapped


def test_fingerprints_match_the_calls_they_describe(monkeypatch, auth: Path) -> None:
    monkeypatch.setattr(client, "_run", fake_run([]))
    backend = client.CodexBackend(binary="codex", auth_file=auth)
    response = backend.complete(client.Request("codex", "gpt-6.1-sol", "high", "S", "P"))
    assert backend.fingerprint("gpt-6.1-sol", "high") == {
        "cli_version": response.summary.cli_version, "isolation": response.summary.isolation}
    claude = client.ClaudeBackend(binary="claude-cli")
    *_, settings = claude.prepare(client.Request("claude", "m", "high", "S", "P"))
    monkeypatch.setattr(claude, "version", lambda: "2.1.9 (Claude Code)")
    assert claude.fingerprint("m", "high")["isolation"] == client.sha256_text(
        client.canonical_json(settings))


def test_setup_failures_are_not_charged_as_calls(tmp_path: Path) -> None:
    base = tmp_path / "a"
    base.mkdir()
    with pytest.raises(client.BackendError) as raised:
        client.CodexBackend(auth_file=tmp_path / "missing.json").prepare(
            base, client.Request("codex", "gpt-6.1-sol", "high", "S", "P"))
    assert raised.value.called is False
    assert client.BackendError("exit 1").called is True


def test_binaries_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CLI_BIN", "my-claude")
    monkeypatch.setenv("CODEX_CLI_BIN", "my-codex")
    assert client.ClaudeBackend().binary == "my-claude"
    assert client.CodexBackend().binary == "my-codex"


def test_private_terms_are_found_case_insensitively() -> None:
    assert client.private_matches("Memo for JANE Q", ["jane q", "", "x@y.z"]) == ["jane q"]
    assert client.private_matches("nothing here", ["jane q"]) == []
