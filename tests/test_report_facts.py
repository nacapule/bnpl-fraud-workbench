"""Case files print their facts only through placeholders: single values in their own
units, tables of records and of named values, and a loud failure on anything missing, so
a case's text cannot drift from its facts file."""

from __future__ import annotations

import json
import shutil

import pytest

from report.render import REPO, RenderError, Sources, out_of_sync, render, render_all

FACTS = {
    "account_takeover": {
        "alerts": {"account_takeover": {
            "selected": True,
            "decision": {"evidence_at": "2025-07-03 15:40:04", "recorded_action": "hold",
                         "analyst_started_at": None},
            "evidence": {"row": {"amount_cents": 26387, "account_age_days": 512.0002,
                                 "attempts_user_24h": 1, "ship_to_home": 0,
                                 "device_link_age_hours": 0.2367},
                         "rules_held": ["R03"]},
            "later": {"prevented_cents": 14514, "differs": True,
                      "incumbent": {"events": [
                          {"event": "checkout", "detail": "processor approved",
                           "known_at": "2025-07-03 15:40:04", "amount_cents": 26387},
                          {"event": "shipped", "detail": None,
                           "known_at": "2025-07-04 04:30:51", "amount_cents": None}]}},
        }},
    },
}
EVENTS = "{{ table:fact:account_takeover:alerts.account_takeover.later.incumbent.events"


@pytest.fixture
def sources() -> Sources:
    return Sources(summary={"metrics": {}, "tables": {}}, facts=FACTS)


def test_a_fact_prints_in_the_unit_of_its_name(sources):
    path = "account_takeover:alerts.account_takeover"
    assert render(f"{{{{ fact:{path}.later.prevented_cents }}}}", sources) == "$145"
    assert render(f"{{{{ fact:{path}.later.prevented_cents | usd:2 }}}}", sources) == "$145.14"
    assert render(f"{{{{ fact:{path}.evidence.row.account_age_days | num:0 }}}}", sources) == "512"
    assert render(f"{{{{ fact:{path}.evidence.row.attempts_user_24h }}}}", sources) == "1"
    assert render(f"{{{{ fact:{path}.decision.recorded_action }}}}", sources) == "hold"
    assert render(f"{{{{ fact:{path}.decision.evidence_at | date }}}}", sources) == "3 July 2025"
    assert render(f"{{{{ fact:{path}.evidence.rules_held.0 }}}}", sources) == "R03"
    with pytest.raises(RenderError, match="format 'pct' does not fit unit 'cents'"):
        render(f"{{{{ fact:{path}.later.prevented_cents | pct }}}}", sources)


@pytest.mark.parametrize("expression, message", [
    ("ring:alerts.ring.later.prevented_cents", "no case facts file 'ring'"),
    ("account_takeover:alerts.account_takeover.later.loss_cents", "has no"),
    ("account_takeover:alerts.account_takeover.evidence.rules_held.3", "has no"),
    ("account_takeover:alerts.account_takeover.evidence.row", "not a single value"),
    ("account_takeover:alerts.account_takeover.decision.analyst_started_at", "is null"),
    ("account_takeover", "is not <file>:<dotted.path>"),
])
def test_a_missing_or_unprintable_fact_fails_with_its_line(sources, expression, message):
    with pytest.raises(RenderError) as raised:
        render(f"Intro.\n{{{{ fact:{expression} }}}}\n", sources)
    assert raised.value.problems[0].startswith("line 2: ")
    assert message in raised.value.problems[0]


def test_a_list_of_records_renders_as_a_table(sources):
    text = f'{EVENTS} | event "Event", known_at "Known" date, amount_cents "Amount" usd }}}}'
    assert render(text, sources) == (
        "| Event | Known | Amount |\n|---|---:|---:|\n"
        "| checkout | 3 July 2025 | $264 |\n| shipped | 4 July 2025 | n/a |")
    with pytest.raises(RenderError, match="no column 'when'"):
        render(f'{EVENTS} | when "When" }}}}', sources)
    with pytest.raises(RenderError, match="not a list of records"):
        render('{{ table:fact:account_takeover:alerts.account_takeover.evidence | x "X" }}',
               sources)


def test_named_values_of_a_record_render_as_a_two_column_table(sources):
    row = "account_takeover:alerts.account_takeover.evidence.row"
    text = (f'{{{{ facts:{row} "Evidence" "At decision" | amount_cents "Amount" usd:2, '
            f'device_link_age_hours "Device first used (hours before)" num:1, '
            f'ship_to_home "Ships home" }}}}')
    assert render(text, sources) == (
        "| Evidence | At decision |\n|---|---:|\n| Amount | $263.87 |\n"
        "| Device first used (hours before) | 0.2 |\n| Ships home | 0 |")
    assert render(f'{{{{ facts:{row} | attempts_user_24h }}}}', sources) == (
        "| Fact | Value |\n|---|---:|\n| attempts_user_24h | 1 |")
    with pytest.raises(RenderError) as raised:
        render(f'{{{{ facts:{row} | amount_cents "A" pct, missing "M" }}}}', sources)
    assert len(raised.value.problems) == 1 and "missing" in str(raised.value) \
        and "'pct' does not fit" in str(raised.value)
    with pytest.raises(RenderError, match="must be on its own line"):
        render(f'Inline {{{{ facts:{row} | amount_cents }}}} here.', sources)


@pytest.mark.parametrize("entry, message", [
    ('analyst_started_at "Started"', "fact 'analyst_started_at' is null"),
    ('recorded_action "Action" nonexistent', "unknown format 'nonexistent'"),
    ('recorded_action "Action" usd', "format 'usd' does not fit"),
])
def test_a_named_value_that_cannot_print_fails(sources, entry, message):
    with pytest.raises(RenderError, match=message):
        render(f"{{{{ facts:account_takeover:alerts.account_takeover.decision | {entry} }}}}",
               sources)


def test_a_yes_or_no_fact_takes_no_format(sources):
    later = "account_takeover:alerts.account_takeover.later"
    assert render(f'{{{{ facts:{later} | differs "Differs" }}}}', sources).endswith(
        "| Differs | yes |")
    with pytest.raises(RenderError, match="unknown format 'nonexistent'"):
        render(f'{{{{ facts:{later} | differs "Differs" nonexistent }}}}', sources)
    with pytest.raises(RenderError, match="takes no format"):
        render(f'{{{{ facts:{later} | differs "Differs" usd }}}}', sources)


def test_the_repository_reads_every_case_facts_file(tmp_path):
    for part in ("experiments/protocol.yaml", "report/claims.yaml"):
        (tmp_path / part).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / part, tmp_path / part)
    shutil.copytree(REPO / "config", tmp_path / "config")
    (tmp_path / "cases" / "facts").mkdir(parents=True)
    (tmp_path / "cases" / "facts" / "ring.json").write_text(json.dumps({"n_count": 3}))
    sources = Sources.from_repo({"metrics": {}, "tables": {}}, root=tmp_path)
    assert sources.facts == {"ring": {"n_count": 3}}
    assert render("{{ fact:ring:n_count }}", sources) == "3"


def test_a_case_whose_facts_change_no_longer_matches_its_rendered_text(tmp_path, sources):
    templates = tmp_path / "report" / "templates"
    (templates / "cases").mkdir(parents=True)
    (templates / "cases" / "CASE-01.md").write_text(
        "Prevented {{ fact:account_takeover:alerts.account_takeover.later.prevented_cents }}.\n")
    render_all(sources, tmp_path, templates, tmp_path)
    assert out_of_sync(sources, tmp_path, templates, tmp_path) == []
    changed = json.loads(json.dumps(FACTS))
    changed["account_takeover"]["alerts"]["account_takeover"]["later"]["prevented_cents"] = 0
    stale = Sources(summary=sources.summary, facts=changed)
    assert out_of_sync(stale, tmp_path, templates, tmp_path) == [
        "cases/CASE-01.md: differs from its template and the results (run python -m report render)"]
    del changed["account_takeover"]["alerts"]["account_takeover"]["later"]["prevented_cents"]
    [problem] = out_of_sync(stale, tmp_path, templates, tmp_path)
    assert "has no 'alerts.account_takeover.later.prevented_cents'" in problem


def test_placeholders_without_facts_render_as_before():
    plain = Sources(summary={"metrics": {}, "tables": {}})
    assert plain.facts == {}
    assert render("No placeholders here.", plain) == "No placeholders here."
    with pytest.raises(RenderError, match="no case facts file"):
        render("{{ fact:account_takeover:alerts }}", plain)
