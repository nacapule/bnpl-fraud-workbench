"""Values from a memo benchmark's results file: a pointer into
``llm/eval/benchmarks/<id>/results.json`` prints a rate with its counts, an interval, a
difference in percentage points or a plain value, and fails loudly on a missing file or
key, a value that is not one number, a null, or a list item named by its position."""

from __future__ import annotations

from pathlib import Path

import pytest

from report.render import BENCHMARKS, RenderError, Sources, render, render_all

RESULTS = {
    "benchmark": "test",
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
    "endpoints": {"primary": {"arms": {"opus": {
        "natural_mix": {"source": "/statistics/arms/opus/complete_pass/natural",
                        "value": 0.9993},
        "unweighted": {"source": "/arms/opus/summary/complete_pass",
                       "value": {"denominator": 200, "numerator": 198, "value": 0.99}},
        "interval": {"source": "/statistics/arms/opus/complete_pass/cluster_bootstrap",
                     "value": [0.9747, 1.0]},
        "stale": {"source": "/statistics/arms/opus/complete_pass/natural", "value": 0.5}}}}},
    "case_memos": [
        {"file": "never_pay_vs_hardship", "slot": "never_pay", "disposition": "decline"},
        {"file": "never_pay_vs_hardship", "slot": "hardship", "disposition": "clear"},
        {"file": "ring", "slot": "ring", "disposition": "escalate"},
    ],
    "bad": {"rate": {"denominator": 200, "numerator": 198, "value": 0.95},
            "counts": {"denominator": 2.5, "numerator": 1, "value": 0.4},
            "share": 1.5, "interval": [0.2, 1.5], "triple": [0.1, 0.2, 0.3]},
}


@pytest.fixture
def sources() -> Sources:
    return Sources(summary={"metrics": {}, "tables": {}}, benchmarks={"test": RESULTS})


def bench(pointer: str, sources: Sources, form: str = "") -> str:
    return render("{{ bench:test:" + pointer + (f" | {form}" if form else "") + " }}", sources)


@pytest.mark.parametrize("pointer, form, text", [
    # a rate prints as a share and with its counts, from its whole counts
    ("/arms/opus/summary/complete_pass", "pct", "99.0%"),
    ("/arms/opus/summary/complete_pass", "of", "198 of 200"),
    ("/arms/opus/summary/complete_pass", "n", "198/200"),
    ("/arms/opus/summary/claim_errors", "pct:3", "0.026%"),
    # an endpoint entry is read at its source
    ("/endpoints/primary/arms/opus/natural_mix", "pct:2", "99.93%"),
    ("/endpoints/primary/arms/opus/unweighted", "of", "198 of 200"),
    ("/endpoints/primary/arms/opus/interval", "bounds:2", "97.47% to 100.00%"),
    # an interval of shares, of differences, a difference, a p-value, counts and text
    ("/statistics/arms/opus/complete_pass/cluster_bootstrap", "bounds", "97.5% to 100.0%"),
    ("/statistics/paired/opus vs sol/complete_pass/difference_cluster_bootstrap", "bounds",
     "-0.5 pp to +4.6 pp"),
    ("/statistics/paired/opus vs sol/complete_pass/difference", "pp", "+2.0 pp"),
    ("/statistics/paired/opus vs sol/complete_pass/cluster_sign_test_p", "num:2", "0.29"),
    ("/statistics/paired/opus vs sol/complete_pass/only_first", "count", "6"),
    ("/arms/opus/disagreement/hold -> needs_check", "", "61"),
    ("/pins/opus/model", "", "a-model"),
    ("/statistics/arms/sol/complete_pass/degenerate", "", "every case passed"),
    # keys holding / and ~ are escaped as ~1 and ~0
    ("/arms/opus/cases/case-1~1primary/outcome", "", "scored"),
    ("/arms/opus/cases/case-1~1primary/complete_pass", "yesno", "yes"),
    ("/arms/opus/cases/a~0b/outcome", "", "format_failure"),
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
    ("/bad/triple", "bounds", "is a list, not a single value"),
    ("", "", "is not <benchmark id>:<JSON pointer>"),
    # a null, with the reason beside it
    ("/statistics/arms/sol/complete_pass/cluster_bootstrap", "bounds",
     r"is null \(every case passed; print its degenerate reason\)"),
    # values that are not what their kind requires
    ("/bad/rate", "pct", "value 0.95 is not 198/200 to four places"),
    ("/bad/counts", "pct", "is not a rate of whole counts"),
    ("/bad/share", "pct", "1.5 is not a share between 0 and 1"),
    ("/bad/interval", "bounds", "1.5 is not a share between 0 and 1"),
    ("/endpoints/primary/arms/opus/stale", "pct", "does not hold the value at its source"),
    # formats that do not fit the kind
    ("/statistics/paired/opus vs sol/complete_pass/difference", "pct",
     "a difference of rates is in percentage points"),
    ("/statistics/paired/opus vs sol/complete_pass/cluster_sign_test_p", "pct",
     "does not fit a number with no declared unit"),
    ("/statistics/paired/opus vs sol/complete_pass/only_first", "pct",
     "does not fit a number with no declared unit"),
    ("/statistics/arms/opus/complete_pass/natural", "bounds", "needs an interval"),
    ("/statistics/arms/opus/complete_pass/cluster_bootstrap", "pct", "is not a number"),
    ("/statistics/arms/opus/complete_pass/natural", "of", "needs a result metric"),
])
def test_a_benchmark_value_that_cannot_mean_what_it_says_fails(sources, pointer, form,
                                                                message):
    with pytest.raises(RenderError, match=message):
        bench(pointer, sources, form)


def test_a_benchmark_without_results_fails(sources):
    with pytest.raises(RenderError, match="no benchmark results 'elsewhere'"):
        render("{{ bench:elsewhere:/benchmark }}", sources)


@pytest.mark.parametrize("placeholder", [
    "bench:test:/statistics/arms/opus/complete_pass/missing | pct",
    "bench:missing:/benchmark",
    "bench:test:/case_memos/1/slot",
    "bench:test:/statistics/arms/sol/complete_pass/cluster_bootstrap | bounds",
    "bench:test:/statistics/arms/opus | pct",
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
