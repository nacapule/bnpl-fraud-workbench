"""A context row's flags (0 or 1) print as yes or no in case files, and nothing that is
not a flag can print that way: a count of 2, a fraction, a missing value, a word, an
amount or a result metric is refused."""

from __future__ import annotations

import pytest

from core.results import Metric
from report import formats
from report.formats import FormatError, Value
from report.render import RenderError, Sources, render

FACTS = {"case": {"row": {"avs_mismatch": 1, "ship_to_home": 0, "attempts_user_24h": 2,
                          "amount_cents": 1, "share": 0.5, "missing": None,
                          "word": "1"},
                  "flags": {"settled": True}}}


@pytest.fixture
def sources() -> Sources:
    return Sources(summary={"metrics": {}, "tables": {}}, facts=FACTS)


@pytest.mark.parametrize("flag, text", [(1, "yes"), (0, "no"), (True, "yes"), (False, "no")])
def test_a_flag_prints_as_yes_or_no(flag, text):
    assert formats.apply("yesno", Value(plain=flag), []) == text


@pytest.mark.parametrize("plain, unit, message", [
    (2, None, "not a flag"),
    (0.5, None, "not a flag"),
    (1.0, None, "not a flag"),
    (None, None, "not a flag"),
    ("1", None, "not a flag"),
    (1, "cents", "does not fit unit 'cents'"),
    (1, "count", "does not fit unit 'count'"),
])
def test_what_is_not_a_flag_is_refused(plain, unit, message):
    with pytest.raises(FormatError, match=message):
        formats.apply("yesno", Value(plain=plain, unit=unit), [])


def test_a_result_metric_or_an_argument_is_refused():
    metric = Metric(value=1, unit="count", population="test-window orders", window="test")
    with pytest.raises(FormatError, match="not result metrics"):
        formats.apply("yesno", Value(metric=metric), [])
    with pytest.raises(FormatError, match="takes no arguments"):
        formats.apply("yesno", Value(plain=1), ["2"])


def test_case_facts_print_their_flags_inline_and_in_tables(sources):
    assert render("{{ fact:case:row.avs_mismatch | yesno }}", sources) == "yes"
    assert render("{{ fact:case:flags.settled | yesno }}", sources) == "yes"
    table = render('{{ facts:case:row "Fact" "Value" | avs_mismatch "AVS failed" yesno, '
                   'ship_to_home "Ships home" yesno }}', sources)
    assert table.splitlines() == ["| Fact | Value |", "|---|---:|", "| AVS failed | yes |",
                                  "| Ships home | no |"]


@pytest.mark.parametrize("entry, message", [
    ("attempts_user_24h", "not a flag"),
    ("share", "not a flag"),
    ("word", "not a flag"),
    ("amount_cents", "does not fit unit 'cents'"),
])
def test_a_case_fact_that_is_not_a_flag_fails_to_render(sources, entry, message):
    with pytest.raises(RenderError, match=message):
        render(f"{{{{ fact:case:row.{entry} | yesno }}}}", sources)
    with pytest.raises(RenderError, match=message):
        render(f'{{{{ facts:case:row | {entry} "Flag" yesno }}}}', sources)
