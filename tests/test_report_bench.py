"""Values from a memo benchmark's results file: a pointer into
``llm/eval/benchmarks/<id>/results.json`` prints a rate with its counts, an interval, a
difference in percentage points or a plain value, each only in the formats its kind
takes, and fails loudly on a missing file or key, a value that is not one number, a
null, a part of a rate, a stale copy of an endpoint, or a list item named by its
position."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from report.render import (
    BENCHMARKS,
    RenderError,
    Sources,
    load_benchmark,
    render,
    render_all,
)

RESULTS = {
    "benchmark": "test",
    "axes": {"check_completed": 0.2151, "review": 0.7849},
    "pins": {"opus": {"model": "a-model", "cli_version": "1.2.3"}},
    "statistics": {
        "arms": {"opus": {"complete_pass": {
            "natural": 0.9993, "rate": 0.99, "cluster_bootstrap": [0.9747, 1.0]}},
                 "sol": {"complete_pass": {
                     "natural": 1.0, "cluster_bootstrap": None,
                     "degenerate": "every case passed"}}},
        "paired": {"opus vs sol": {"complete_pass": {
            "difference": 0.02, "difference_cluster_bootstrap": [-0.0051, 0.0464],
            "cluster_sign_test_p": 0.289062, "only_first": 6}}},
    },
    "arms": {"opus": {"summary": {"complete_pass": {"denominator": 200, "numerator": 198,
                                                     "value": 0.99},
                                   "claim_errors": {"denominator": 3843, "numerator": 1,
                                                    "value": 0.0003}},
                      "disagreement": {"hold -> needs_check": 61},
                      "cases": {"case-1/primary": {"outcome": "scored", "complete_pass": True},
                                "a~b": {"outcome": "format_failure"}}}},
    "endpoints": {
        "primary": {"arms": {"opus": {
            "natural_mix": {"source": "/statistics/arms/opus/complete_pass/natural",
                            "value": 0.9993},
            "unweighted": {"source": "/arms/opus/summary/complete_pass",
                           "value": {"denominator": 200, "numerator": 198, "value": 0.99}},
            "interval": {"source": "/statistics/arms/opus/complete_pass/cluster_bootstrap",
                         "value": [0.9747, 1.0]},
            "stale": {"source": "/statistics/arms/opus/complete_pass/natural",
                      "value": 0.5}}}},
        "paired": {"source": "/statistics/paired", "value": {"opus vs sol": {
            "complete_pass": {"difference": 0.5}}}},
        "loop": {"source": "/endpoints/loop/value", "value": 1}},
    "case_memos": [
        {"file": "never_pay_vs_hardship", "slot": "never_pay", "disposition": "decline",
         "claims": [{"field": "context.account_age_days", "value": 0.0243},
                    {"field": "context.claimed_days", "value": 12}]},
        {"file": "never_pay_vs_hardship", "slot": "hardship", "disposition": "clear"},
        {"file": "ring", "slot": "ring", "disposition": "escalate"},
        {"file": "a/b", "slot": "x", "disposition": "hold"},
    ],
    "bad": {"rate": {"denominator": 200, "numerator": 198, "value": 0.95},
            "counts": {"denominator": 2.5, "numerator": 1, "value": 0.4},
            "natural": 1.5, "difference": 1.5, "wilson": [0.2, 1.5],
            "cluster_bootstrap": [0.9, 0.1], "difference_cluster_bootstrap": [2.0, 3.0],
            "pass_share_interval": [0.1, 0.2, 0.3], "pair": [0.1, 0.2], "speed": 0.25,
            "cluster_sign_test_p": 1.5},
}


@pytest.fixture
def sources() -> Sources:
    return Sources(summary={"metrics": {}, "tables": {}}, benchmarks={"test": RESULTS})


def bench(pointer: str, sources: Sources, form: str = "") -> str:
    return render("{{ bench:test:" + pointer + (f" | {form}" if form else "") + " }}", sources)


@pytest.mark.parametrize("pointer, form, text", [
    # a rate prints as a share and with its counts, from its whole counts
    ("/arms/opus/summary/complete_pass", "", "99.0%"),
    ("/arms/opus/summary/complete_pass", "of", "198 of 200"),
    ("/arms/opus/summary/complete_pass", "n", "198/200"),
    ("/arms/opus/summary/complete_pass", "numerator", "198"),
    ("/arms/opus/summary/claim_errors", "pct:3", "0.026%"),
    # an endpoint entry, or a path through its value, is read at its source
    ("/endpoints/primary/arms/opus/natural_mix", "pct:2", "99.93%"),
    ("/endpoints/primary/arms/opus/natural_mix/value", "pct:2", "99.93%"),
    ("/endpoints/primary/arms/opus/unweighted", "of", "198 of 200"),
    ("/endpoints/primary/arms/opus/interval", "bounds:2", "97.47% to 100.00%"),
    ("/endpoints/primary/arms/opus/natural_mix/source", "", "/statistics/arms/opus/"
     "complete_pass/natural"),
    # shares, intervals of shares and of differences, a difference, a p-value, counts, text
    ("/axes/check_completed", "pct", "21.5%"),
    ("/statistics/arms/opus/complete_pass/cluster_bootstrap", "bounds", "97.5% to 100.0%"),
    ("/statistics/paired/opus vs sol/complete_pass/difference_cluster_bootstrap", "bounds",
     "-0.5 pp to +4.6 pp"),
    ("/statistics/paired/opus vs sol/complete_pass/difference", "", "+2.0 pp"),
    ("/statistics/paired/opus vs sol/complete_pass/cluster_sign_test_p", "num:2", "0.29"),
    ("/statistics/paired/opus vs sol/complete_pass/only_first", "", "6"),
    ("/arms/opus/disagreement/hold -> needs_check", "", "61"),
    ("/pins/opus/model", "", "a-model"),
    ("/statistics/arms/sol/complete_pass/degenerate", "", "every case passed"),
    # a claim's field is text
    ("/case_memos/[slot=never_pay]/claims/[field=context.account_age_days]/field", "",
     "context.account_age_days"),
    # keys holding / and ~ are escaped as ~1 and ~0, in selectors too
    ("/arms/opus/cases/case-1~1primary/outcome", "", "scored"),
    ("/arms/opus/cases/case-1~1primary/complete_pass", "", "yes"),
    ("/arms/opus/cases/a~0b/outcome", "", "format_failure"),
    ("/case_memos/[file=a~1b]/disposition", "", "hold"),
    # a list item by its fields
    ("/case_memos/[file=never_pay_vs_hardship,slot=hardship]/disposition", "", "clear"),
    ("/case_memos/[slot=ring]/disposition", "", "escalate"),
])
def test_a_benchmark_value_prints_in_its_kind(sources, pointer, form, text):
    assert bench(pointer, sources, form) == text


@pytest.mark.parametrize("pointer, form, message", [
    # the file, the key, the pointer
    ("/statistics/arms/nobody/complete_pass/rate", "", "has no /statistics/arms/nobody"),
    ("statistics/arms", "", "is not <benchmark id>:<JSON pointer>"),
    ("", "", "is not <benchmark id>:<JSON pointer>"),
    ("/arms/opus/cases/a~2b/outcome", "", "~ is escaped as ~0 and / as ~1"),
    ("/arms/opus/cases/case-1/primary/outcome", "", "has no /arms/opus/cases/case-1"),
    ("/pins/opus/model/name", "", "is a single value, not a container"),
    # a list item named by its position, or by fields that do not name one item
    ("/case_memos/0/disposition", "", "select an item by its fields"),
    ("/case_memos/[file=never_pay_vs_hardship]/disposition", "", "selects 2 items"),
    ("/case_memos/[file=traveller]/disposition", "", "selects 0 items"),
    ("/case_memos/[file]/disposition", "", "each selector item is key=value"),
    # not one value
    ("/statistics/arms/opus", "", "is not a single value; point at one of"),
    ("/case_memos", "", "is a list, not a single value"),
    ("/bad/pair", "", "is a list, not a single value"),
    ("/bad/pass_share_interval", "bounds", "is not an interval: a pair of numbers"),
    # a null, with the reason beside it
    ("/statistics/arms/sol/complete_pass/cluster_bootstrap", "bounds",
     r"is null \(every case passed; print its degenerate reason\)"),
    # a rate is whole: its parts cannot be printed on their own
    ("/arms/opus/summary/claim_errors/value", "", "is a rate; it prints whole"),
    ("/endpoints/primary/arms/opus/unweighted/value/value", "", "is a rate; it prints whole"),
    ("/bad/rate", "pct", "value 0.95 is not 198/200 to four places"),
    ("/bad/counts", "pct", "is not a rate of whole counts"),
    # an endpoint's copy that is not its source's value, however it is reached
    ("/endpoints/primary/arms/opus/stale", "pct", "does not hold the value at its source"),
    ("/endpoints/primary/arms/opus/stale/value", "pct", "does not hold the value"),
    ("/endpoints/paired/value/opus vs sol/complete_pass/difference", "",
     "does not hold the value at its source /statistics/paired"),
    ("/endpoints/loop", "", "form a loop"),
    # values out of their range, intervals out of order
    ("/bad/natural", "pct", "1.5 is not a share between 0 and 1"),
    ("/bad/difference", "pp", "1.5 is not a difference of shares between -1 and 1"),
    ("/bad/wilson", "bounds", "1.5 is not a share between 0 and 1"),
    ("/bad/cluster_bootstrap", "bounds", "bounds are out of order"),
    ("/bad/difference_cluster_bootstrap", "bounds", "2.0 is not a difference of shares"),
    # formats that do not fit the kind
    ("/statistics/paired/opus vs sol/complete_pass/difference", "pct",
     "a benchmark difference prints with pp, not pct"),
    ("/statistics/paired/opus vs sol/complete_pass/difference", "num",
     "a benchmark difference prints with pp, not num"),
    ("/statistics/arms/opus/complete_pass/natural", "num", "a benchmark share prints with pct"),
    ("/statistics/paired/opus vs sol/complete_pass/cluster_sign_test_p", "pct",
     "a benchmark number prints with count, num, not pct"),
    # a number whose unit the results do not state, and any claim's value
    ("/case_memos/[slot=never_pay]/claims/[field=context.account_age_days]/value", "num",
     "a claim's value has no unit the results state"),
    ("/case_memos/[slot=never_pay]/claims/[field=context.claimed_days]/value", "",
     "a claim's value has no unit the results state"),
    ("/bad/speed", "num", "0.25 has no unit the results state"),
    ("/bad/cluster_sign_test_p", "num", "1.5 is not a p-value between 0 and 1"),
    ("/statistics/arms/opus/complete_pass/natural", "bounds", "prints with pct, not bounds"),
    ("/statistics/arms/opus/complete_pass/cluster_bootstrap", "pct",
     "a benchmark interval prints with bounds, not pct"),
    ("/arms/opus/summary/complete_pass", "num", "a benchmark rate prints with pct, of"),
    ("/pins/opus/model", "count", "a benchmark text prints with value, not count"),
    ("/arms/opus/cases/case-1~1primary/complete_pass", "count",
     "a benchmark flag prints with yesno, not count"),
])
def test_a_benchmark_value_that_cannot_mean_what_it_says_fails(sources, pointer, form,
                                                                message):
    with pytest.raises(RenderError, match=message):
        bench(pointer, sources, form)


@pytest.mark.parametrize("placeholder", [
    "bench:test:/statistics/arms/opus/complete_pass/natural|pct",
    "bench:test:/statistics/arms/opus/complete_pass/natural | pct | num",
])
def test_a_pipe_belongs_to_the_format_alone(sources, placeholder):
    with pytest.raises(RenderError, match="a benchmark pointer cannot hold |"):
        render("{{ " + placeholder + " }}", sources)


def test_a_benchmark_without_results_fails(sources):
    with pytest.raises(RenderError, match="no benchmark results 'elsewhere'"):
        render("{{ bench:elsewhere:/benchmark }}", sources)


def test_a_results_file_with_a_number_json_does_not_allow_fails(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text('{"rate": {"denominator": 200, "numerator": 198, "value": NaN}}')
    with pytest.raises(ValueError, match="NaN is not a JSON number"):
        load_benchmark(path)
    path.write_text(json.dumps({"rate": 0.5}))
    assert load_benchmark(path) == {"rate": 0.5}


@pytest.mark.parametrize("placeholder", [
    "bench:test:/statistics/arms/opus/complete_pass/missing | pct",
    "bench:missing:/benchmark",
    "bench:test:/case_memos/1/slot",
    "bench:test:/statistics/arms/sol/complete_pass/cluster_bootstrap | bounds",
    "bench:test:/statistics/arms/opus | pct",
    "bench:test:/arms/opus/summary/claim_errors/value | pct",
    "bench:test:/endpoints/primary/arms/opus/stale/value | pct",
    "bench:test:/statistics/paired/opus vs sol/complete_pass/difference | num",
    "bench:test:/bad/cluster_bootstrap | bounds",
])
def test_the_repository_render_fails_on_a_benchmark_value_it_cannot_print(
        tmp_path: Path, sources: Sources, placeholder: str) -> None:
    templates = tmp_path / "report" / "templates"
    templates.mkdir(parents=True)
    (templates / "README.md").write_text("Rate: {{ " + placeholder + " }}.\n")
    with pytest.raises(RenderError, match="line 1"):
        render_all(sources, tmp_path, templates, tmp_path)
    assert not (tmp_path / "README.md").exists()
    (templates / "README.md").write_text(
        "Rate: {{ bench:test:/arms/opus/summary/complete_pass | of }}.\n")
    render_all(sources, tmp_path, templates, tmp_path)
    assert (tmp_path / "README.md").read_text().endswith("Rate: 198 of 200.\n")


def test_the_repository_reads_every_benchmark_results_file() -> None:
    root = Path(__file__).resolve().parents[1]
    loaded = Sources.from_repo({"metrics": {}, "tables": {}}).benchmarks
    assert set(loaded) == {path.parent.name
                           for path in (root / BENCHMARKS).glob("*/results.json")}
