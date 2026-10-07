"""Rendered documents: formats, templates, the claims behind directional sentences, the lint."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core import protocol as proto
from core.results import Interval, Metric, SeedSpread, StageResult, assemble_summary, metric
from report import claims as claims_module
from report import formats, lint
from report.__main__ import check as check_repository
from report.__main__ import main
from report.formats import FormatError, Value
from report.render import (
    REPO,
    RenderError,
    Sources,
    documents,
    out_of_sync,
    render,
    render_all,
)

NET_SEEDS = [150_000, 90_000, 200_000, -20_000, 130_000, 110_000, 95_000, 140_000, 120_000,
             219_560]


def rate(numerator: int, denominator: int, **extra) -> Metric:
    return Metric.from_ratio(numerator, denominator, population="memos scored",
                             window="test", **extra)


def cents(value: float, seeds: dict[int, float] | None = None) -> Metric:
    return Metric(value=value, unit="cents", population="net cash, test-window orders",
                  window="test", seeds=None if seeds is None else SeedSpread(seeds))


def count(value: int) -> Metric:
    return Metric(value=value, unit="count", population="cases", window="test")


def summary() -> dict:
    one_seed = {1: 1_000_000, **{seed: 0 for seed in range(2, 11)}}
    metrics = {
        "llm.accuracy.v1": rate(60, 200),
        "llm.accuracy.v2": rate(85, 200),
        "llm.consistency": rate(42, 50, interval=Interval(0.71, 0.92, "wilson")),
        "llm.accuracy.vs_v1.v2": Metric(value=1250, unit="bps", population="v2 minus v1",
                                        window="test", interval=Interval(400, 2100, "exact")),
        "llm.accuracy.vs_v1.v3": Metric(value=50, unit="bps", population="v3 minus v1",
                                        window="test", interval=Interval(-300, 400, "exact")),
        "llm.right.vs_v1.v2.only_policy": count(32),
        "llm.right.vs_v1.v2.only_reference": count(5),
        "evaluate.net.vs_incumbent.surge.high.hybrid": cents(
            123_456, dict(enumerate(NET_SEEDS, start=1))),
        "replay.net.hybrid": cents(9e5, dict(enumerate([9e5] * 10, start=1))),
        "replay.net.vs_incumbent.hybrid": cents(123_456, dict(enumerate(NET_SEEDS, start=1))),
        "replay.net.vs_incumbent.flat": cents(0.0, {1: 500, 2: -400, 3: 300, 4: -100, 5: -300}),
        "replay.net.vs_incumbent.one_seed": cents(1e5, one_seed),
        "replay.missing": Metric.not_evaluated(unit="rate", population="never-pay orders",
                                               window="test", reason="no orders",
                                               numerator=0, denominator=0),
        "replay.withheld": Metric.not_evaluated(unit="rate", population="never-pay orders",
                                                window="test", reason="below minimum support",
                                                numerator=43, denominator=55),
        "replay.mean_cases": Metric(value=2.5, unit="count", population="cases", window="test"),
        "replay.only_first": count(32),
        "replay.only_second": count(5),
        "replay.mean_first": Metric(value=6.0, unit="count", population="cases", window="test"),
        "replay.mean_second": Metric(value=0.9, unit="count", population="cases",
                                     window="test"),
        "replay.seed_mean_first": Metric(value=6, unit="count", population="cases",
                                         window="test", seeds=SeedSpread(
                                             {seed: 6 for seed in range(1, 11)})),
        "replay.seed_mean_second": Metric(value=2, unit="count", population="cases",
                                          window="test", seeds=SeedSpread(
                                              {seed: 2 for seed in range(1, 11)})),
    }
    stage = StageResult(
        stage="replay",
        versions={"world": "w1"},
        inputs={},
        metrics=metrics,
        tables={"replay.policies": [
            {"policy": "approve_all", "net_cents": 0, "held_share": 0.0, "evaluated": True,
             "held_bps": 50, "seeds": 10, "used_share_vs_rules_bps": 0},
            {"policy": "hybrid", "net_cents": 1_234_567, "held_share": 0.0123,
             "evaluated": False, "held_bps": 12.5, "seeds": 10, "used_share_vs_rules_bps": 2500},
        ]},
    )
    return assemble_summary([stage])


@pytest.fixture
def sources() -> Sources:
    return Sources(
        summary=summary(),
        protocol={"windows": {"test": {"start": "2025-06-01"}}, "seeds": {"canonical": 416}},
        configs={"world": {"product": {"down_payment_bps": 2500, "terms": 3}}},
    )


def value_of(key: str) -> Value:
    return Value(metric=metric(summary(), key), contrast=".vs_" in key)


# ---------------------------------------------------------------- formats
def test_a_change_in_rates_is_percentage_points_not_percent(sources: Sources) -> None:
    """30.0% to 42.5% is +12.5 percentage points; printing it as "+12.5%" misstates it."""
    assert render("{{ llm.accuracy.v2 - llm.accuracy.v1 | pp }}", sources) == "+12.5 pp"
    assert render("{{ llm.accuracy.v2 - llm.accuracy.v1 }}", sources) == "+12.5 pp"
    assert render("{{ llm.accuracy.vs_v1.v2 | pp }}", sources) == "+12.5 pp"
    for template in ("{{ llm.accuracy.v2 - llm.accuracy.v1 | pct }}",
                     "{{ llm.accuracy.vs_v1.v2 | bps_pct }}",
                     "{{ llm.consistency | pp }}"):
        with pytest.raises(RenderError):
            render(template, sources)


def test_rates_show_their_own_numerator_and_denominator(sources: Sources) -> None:
    text = render("{{ llm.consistency }} ({{ llm.consistency | n }}, {{ llm.consistency | ci }})",
                  sources)
    assert text == "84.0% (42/50, 95% CI 71.0% to 92.0%)"


def test_money_counts_and_signs() -> None:
    loss = Value(metric=Metric(value=-123_456, unit="cents", population="p", window="test"))
    assert formats.apply("usd", loss, []) == "-$1,235"
    assert formats.apply("usd", loss, ["2"]) == "-$1,234.56"
    small = Value(metric=Metric(value=-4, unit="cents", population="p", window="test"))
    assert formats.apply("usd", small, ["signed"]) == "$0"  # no sign on a value that rounds to 0
    gain = Value(metric=Metric(value=50_000, unit="cents", population="p", window="test"))
    assert formats.apply("usd", gain, ["signed"]) == "+$500"
    assert formats.apply("value", Value(metric=count(12_345)), []) == "12,345"
    held = Value(metric=Metric(value=37.25, unit="bps", population="p", window="test"))
    assert formats.apply("per_10k", held, []) == "37.2 per 10,000"
    assert formats.apply("bps", held, ["2"]) == "37.25 bps"


def test_a_mean_of_counts_renders(sources: Sources) -> None:
    assert render("{{ replay.mean_cases }}", sources) == "2.5"


def test_formats_refuse_the_wrong_unit() -> None:
    cases = [
        ("usd", value_of("llm.consistency")),
        ("pct", value_of("replay.net.hybrid")),
        ("bps_pct", value_of("llm.consistency")),  # 84% is not 0.84 bps
        ("pct", Value(plain=0.5)),  # a number with no declared unit
        ("usd", Value(plain=1_000)),
        ("pp", Value(plain=0.1)),
    ]
    for name, resolved in cases:
        with pytest.raises(FormatError):
            formats.apply(name, resolved, [])
    with pytest.raises(FormatError, match="unknown format"):
        formats.apply("percent", value_of("llm.consistency"), [])
    with pytest.raises(FormatError, match="unknown format argument"):
        formats.apply("pct", value_of("llm.consistency"), ["wide"])


def test_unevaluated_metrics_say_so_with_their_n(sources: Sources) -> None:
    assert render("{{ replay.missing }}", sources) == "not evaluated (N=0)"
    assert render("{{ replay.withheld | pct }}", sources) == "not evaluated (N=55)"
    assert render("{{ replay.withheld | n }}", sources) == "43/55"
    assert render("{{ replay.missing | define }}", sources) == "never-pay orders (test window)"


def test_seed_replication_is_reported_with_mean_range_and_sign_count(sources: Sources) -> None:
    text = render("{{ replay.net.vs_incumbent.hybrid | seeds }} "
                  "({{ replay.net.vs_incumbent.hybrid | p }})", sources)
    assert text == "mean +$1,235, min -$200, max +$2,196; positive on 9/10 seeds (p = 0.021)"
    assert render("{{ replay.net.vs_incumbent.flat | signs }}", sources) == "negative on 3/5 seeds"
    tie = Value(metric=Metric(value=0.0, unit="cents", population="p", window="test",
                              seeds=SeedSpread({1: 5, 2: -5})))
    assert formats.apply("signs", tie, []) == "mixed: 1 positive and 1 negative of 2 seeds"


def test_fixed_definitions_from_the_protocol_and_configuration(sources: Sources) -> None:
    assert render("{{ protocol:windows.test.start | date }}", sources) == "1 June 2025"
    assert render("{{ protocol:seeds.canonical }}", sources) == "416"
    assert render("{{ config:world:product.down_payment_bps | bps_pct }}", sources) == "25%"
    with pytest.raises(RenderError, match="no declared unit"):  # terms: a plain count
        render("{{ config:world:product.terms | bps_pct }}", sources)


# ---------------------------------------------------------------- templates
def test_rendering_fails_on_every_missing_key_with_its_line(sources: Sources) -> None:
    template = "Accuracy {{ llm.accuracy.v3 }}.\nNet {{ replay.net.nobody | usd }}.\n"
    with pytest.raises(RenderError) as raised:
        render(template, sources, "README.md")
    problems = raised.value.problems
    assert len(problems) == 2
    assert "line 1" in problems[0] and "llm.accuracy.v3" in problems[0]
    assert "line 2" in problems[1] and "replay.net.nobody" in problems[1]


def test_protocol_and_configuration_paths_must_exist(sources: Sources) -> None:
    for template in ("{{ protocol:windows.final.start }}", "{{ config:policy:bands.review }}",
                     "{{ protocol:windows.test }}"):
        with pytest.raises(RenderError):
            render(template, sources)


def test_generated_tables_take_units_from_column_names(sources: Sources) -> None:
    template = ('Before\n{{ table:replay.policies | policy "Policy", net_cents "Net" usd, '
                'held_share "Held" pct:2, evaluated, held_bps "Held" per_10k }}\nAfter')
    assert render(template, sources).split("\n") == [
        "Before",
        "| Policy | Net | Held | evaluated | Held |",
        "|---|---:|---:|---|---:|",
        "| approve_all | $0 | 0.00% | yes | 50.0 per 10,000 |",
        "| hybrid | $12,346 | 1.23% | no | 12.5 per 10,000 |",
        "After",
    ]
    with pytest.raises(RenderError, match="own line"):
        render("See {{ table:replay.policies | policy }} here", sources)
    with pytest.raises(RenderError) as raised:  # every problem, not the first
        render("{{ table:replay.policies | policy, missing_one, missing_two, held_bps pct }}",
               sources)
    message = str(raised.value)
    assert "missing_one" in message and "missing_two" in message
    assert "does not fit unit 'bps'" in message  # 50 bps is not 5,000%
    with pytest.raises(RenderError, match="no declared unit"):
        render("{{ table:replay.policies | seeds pct }}", sources)
    # a paired difference in a table prints as points, never as a percent
    assert render("{{ table:replay.policies | used_share_vs_rules_bps pp }}",
                  sources).split("\n")[2:] == ["| 0.0 pp |", "| +25.0 pp |"]
    for form in ("bps_pct", "pct"):
        with pytest.raises(RenderError):
            render(f"{{{{ table:replay.policies | used_share_vs_rules_bps {form} }}}}", sources)


def _repo(tmp_path: Path, template: str) -> tuple[Path, Path]:
    templates = tmp_path / "report" / "templates"
    (templates / "cases").mkdir(parents=True)
    (templates / "cases" / "CASE-01.md").write_text(template)
    return tmp_path, templates


def test_documents_render_from_templates_and_drift_is_detected(
    tmp_path: Path, sources: Sources
) -> None:
    root, templates = _repo(tmp_path, "# Case\n\nAccuracy {{ llm.accuracy.v2 }}.\n")
    assert [doc.name for doc in documents(templates)] == ["cases/CASE-01.md"]
    [path] = render_all(sources, root, templates, root)
    assert path.read_text().endswith("# Case\n\nAccuracy 42.5%.\n")
    assert path.read_text().startswith("<!-- Generated by `python -m report render` from "
                                       "report/templates/cases/CASE-01.md.")
    assert out_of_sync(sources, root, templates, root) == []
    path.write_text(path.read_text().replace("42.5%", "43%"))  # an edit by hand
    assert out_of_sync(sources, root, templates, root) == [
        "cases/CASE-01.md: differs from its template and the results (run python -m report render)"
    ]
    path.unlink()
    assert "not rendered" in out_of_sync(sources, root, templates, root)[0]


def test_published_documents_render_only_from_the_committed_results(tmp_path: Path) -> None:
    other = tmp_path / "summary.json"
    other.write_text("{}")
    assert main(["render", "--summary", str(other)]) == 1
    assert main(["render", "--summary", str(other), "--out", str(REPO)]) == 1


# ---------------------------------------------------------------- lint
def test_numbers_typed_into_a_template_are_flagged() -> None:
    template = (
        "The hybrid policy saved 12% of loss across 10 seeds.\n"
        "Rendered: {{ replay.net.hybrid | usd }} and {{ replay.net.hybrid | seeds }}.\n"
        "## 3 findings\n"
        "See [42 orders](x.md) and **7%** of them[^1].\n"
        "[^1]: Accuracy is 84%.\n"
        "[^note]: Seed 416 only.\n"
        "Clause §84% and R01.5 are not identifiers.\n"
        "Nor are R84%, Q84%, FP-84%, P3% or CASE-84%.\n"
        "Nor §84(a)%, §6.3(a)% or §6.3(a)(b)%, though §6.3(a)(b) is.\n"
    )
    findings = lint.number_findings("README.md", template)
    assert [(finding.line, finding.text) for finding in findings] == [
        (1, "12"), (1, "10"), (3, "3"), (4, "42"), (4, "7"), (5, "84"), (6, "416"),
        (7, "84"), (7, "01.5"), (8, "84"), (8, "84"), (8, "84"), (8, "3"), (8, "84"),
        (9, "84"), (9, "6.3"), (9, "6.3"),
    ]


def test_inline_code_with_three_backticks_does_not_open_a_fence() -> None:
    template = "```make final``` runs it.\n\nThe loss was 84% of GMV.\n```\ncode 12\n```\n"
    assert [f.text for f in lint.number_findings("README.md", template)] == ["84"]


@pytest.mark.parametrize("template", [
    "```markdown\n> ```text\n> example 12\n> ```\n```\n\nLoss was 84% of GMV.\n",  # quoted
    "- ```\n  code 12\nLoss was 84% of GMV.\n",  # the list item, and its fence, ended
    "> ```\n> code 12\n\nLoss was 84% of GMV.\n",  # the quote, and its fence, ended
    "- Example:\n\n  ```text\n  code 12\n    ```\n\nLoss was 84% of GMV.\n",  # in an item
    "> ```markdown\n> > ```\n> example 12\n> ```\n>\n> Loss was 84% of GMV.\n",  # a > in code
    "- - ```text\n    code 12\n    ```\n\n    Loss was 84% of GMV.\n",  # two list marks
    "    ```text\n    example 12\n    ```\n\nLoss was 84% of GMV.\n",  # indented code, no fence
    "- > ```text\n  > example 12\n  > ```\n  >\n  > Loss was 84% of GMV.\n",  # a quote in an item
    "- Item.\n\n        code 12\n\nLoss was 84% of GMV.\n",  # indented code in an item
    "- > > ```text\n  > > example 12\n  > > ```\n  > >\n  > > Loss was 84% of GMV.\n",
    "Method\n===\n    example 12\n\nLoss was 84% of GMV.\n",  # code after a setext heading
    "<!-- a note on 12 -->\nText <!-- and 12 --> here.\n\nLoss was 84% of GMV.\n",  # comments
])
def test_a_fence_ends_with_its_closer_or_its_container(template: str) -> None:
    """Fence content is read as content first; a fence never outlives its container."""
    assert [f.text for f in lint.number_findings("README.md", template)] == ["84"]


@pytest.mark.parametrize("template", [
    "- > Context.\n  >\n  > Method.\n\n    Loss was 84% of GMV.\n",  # a list paragraph
    "-\n    First paragraph.\n\n    Loss was 84% of GMV.\n",  # an item opened by a bare mark
    "> Context\nLoss was 84% of GMV.\n",  # a quoted paragraph continued without its mark
    "1. First.\n2. Loss was 84% of GMV.\n",  # the next item, not a continuation
    "- Item.\n\n  Loss was 84% of GMV.\n",
    "| a | b |\n|---|---|\n| Loss was 84% of GMV. | x |\n",
    "> Method.\n    ```text\n    example\n    ```\n    Loss was 84% of GMV.\n",  # lazy, not code
    "Context. <!-- editorial\nnote -->\n    Loss was 84% of GMV.\n",  # a comment inside text
    "`<!--` Loss was 84% of GMV. `-->`\n",  # comment marks inside code spans
    "<!--\n```\n-->\nLoss was 84% of GMV.\n",  # a fence mark inside a comment block
    "Text <!-- never closed, so shown\nLoss was 84% of GMV.\n",
    "<!-- check the denominator --> Loss was 84% of GMV.\n",  # text after a comment block
    "<!-- check\nthe denominator --> Loss was 84% of GMV.\n",
    "Use ``a`b``. Loss was 84% of GMV. See ``c`d``.\n",  # code spans of two backticks
    "[ref]: target\n===\n    Loss was 84% of GMV.\n",  # no heading text, so no heading
])
def test_visible_text_is_never_taken_for_code(template: str) -> None:
    """Container marks are read as CommonMark reads them, so prose stays prose."""
    assert [f.text for f in lint.number_findings("README.md", template)] == ["84"]
    assert [f.text.split(" in ")[0] for f in lint.directional_findings(
        "README.md", template.replace("Loss was", "Loss was more,"), ["more"])] == ["'more'"]


def test_identifiers_code_links_and_list_markers_are_not_numbers() -> None:
    template = (
        "1. Rule R01 and R11 fire; FP-2 §6.3(a) and §5.2 apply at P0 to P3.\n"
        "2. See CASE-01, query Q12, `python pipeline.py run --scale 0.1` and "
        "[the policy](policy/fraud-policy.md#62-rules) on pay-in-4 plans.\n"
        "<!-- a comment with 2025 -->\n"
        "```\nmake final  # 30 worlds\n```\n"
        "[ref]: https://example.com/2025/report\n"
    )
    assert lint.number_findings("README.md", template, ["pay-in-4"]) == []
    assert [f.text for f in lint.number_findings("README.md", template)] == ["4"]


def test_comparisons_typed_into_a_template_are_flagged() -> None:
    """A comparison is published only as a claim written from the results."""
    template = (
        "# Results\n\n"
        "The hybrid policy earned more than the incumbent rules.\n"
        "{{ claim:hybrid-net }} Placed claims are not typed.\n"
        "A lower threshold holds more orders\nfor review.\n"
        "`--fewer` and <!-- better --> and [higher](more.md) and\n"
        "```\nmake best\n```\n"
    )
    words = ["more", "lower", "better", "fewer", "higher", "best"]
    findings = lint.directional_findings("README.md", template, words)
    assert [(f.line, f.text.split(" in ")[0]) for f in findings] == [
        (3, "'more'"), (5, "'lower'"), (5, "'more'"), (7, "'higher'")]
    allowed = ["A lower threshold holds more orders for review."]
    assert [f.line for f in lint.directional_findings("README.md", template, words,
                                                      allowed_sentences=allowed)] == [3, 7]
    # an allowed sentence exempts itself whole, and nothing longer
    for text, lines in [
        ("Intro. A lower threshold holds more orders for review. Next.\n", []),
        ("Result: {{ claim:hybrid-net }} A lower threshold holds more orders for review.\n", []),
        ("A lower threshold holds more orders for review than the incumbent rules.\n", [1, 1]),
        ("So A lower threshold holds more orders for review.\n", [1, 1]),
    ]:
        assert [f.line for f in lint.directional_findings(
            "README.md", text, words, allowed_sentences=allowed)] == lines
    with pytest.raises(ValueError, match="full stop"):
        lint.directional_findings("README.md", "x", words, allowed_sentences=["No stop"])


def test_the_lint_configuration_is_text(tmp_path: Path) -> None:
    """YAML reads a bare no or yes as a boolean, which would drop the word."""
    assert "more" in lint.load_config()["comparatives"]
    bare = tmp_path / "lint.yaml"
    bare.write_text("comparatives: [more, yes]\n")
    with pytest.raises(ValueError, match="quote them"):
        lint.load_config(bare)


def test_published_documents_need_templates(tmp_path: Path) -> None:
    root, templates = _repo(tmp_path, "text\n")
    (root / "cases").mkdir()
    (root / "cases" / "CASE-01.md").write_text("rendered\n")
    (root / "cases" / "CASE-02.md").write_text("by hand\n")
    (root / "README.md").write_text("by hand\n")
    (root / "reports").mkdir()
    (root / "reports" / "model.md").write_text("by hand\n")
    config = {
        "documents": ["README.md", "cases/*.md", "reports/model.md"],
        "not_yet_templated": ["README.md", "cases/CASE-05-traveler-cleared.md",
                              "reports/model.md"],
    }
    assert lint.missing_templates(config, root, templates) == [
        "cases/CASE-02.md: published document without a template",
        "cases/CASE-05-traveler-cleared.md: listed in not_yet_templated but does not exist",
        "reports/model.md: not_yet_templated may only shrink; write a template",
    ]


# ---------------------------------------------------------------- claims
def check(**fields) -> claims_module.Check:
    base = {"test": "sign", "direction": "positive", "key": "replay.net.vs_incumbent.hybrid"}
    base.update(fields)
    return claims_module.Check(**base)


def test_a_supported_comparison_holds_and_a_reversed_one_fails() -> None:
    results = summary()
    assert claims_module.support(check(), results) is None
    assert "not negative" in claims_module.support(check(direction="negative"), results)
    assert "significant" in claims_module.support(check(direction="no_detected_difference"),
                                                  results)
    flat = "replay.net.vs_incumbent.flat"
    assert "not positive" in claims_module.support(check(key=flat), results)
    assert claims_module.support(check(key=flat, direction="no_detected_difference"),
                                 results) is None


def test_a_comparison_needs_a_paired_difference_not_a_level() -> None:
    """Ten seeds of a policy's own net are not evidence that it beat another policy."""
    reason = claims_module.support(check(key="replay.net.hybrid"), summary())
    assert reason is not None and "paired difference" in reason


def test_a_majority_of_seeds_is_not_enough_without_the_test() -> None:
    """Four of five seeds positive is p = 0.375: not a directional result at 0.05."""
    item = Metric(value=1.0, unit="cents", population="p", window="test",
                  seeds=SeedSpread({1: 5, 2: 3, 3: 2, 4: 1, 5: -1}))
    stage = StageResult(stage="replay", versions={}, inputs={}, metrics={"replay.vs_a.x": item})
    reason = claims_module.support(check(key="replay.vs_a.x"), assemble_summary([stage]))
    assert reason is not None and "p = 0.3750" in reason


def test_no_detected_difference_is_not_equivalence() -> None:
    """One seed far off and nine at zero: the test is silent, but the policies are not equal."""
    results = summary()
    one_seed = "replay.net.vs_incumbent.one_seed"
    assert claims_module.support(check(key=one_seed, direction="no_detected_difference"),
                                 results) is None
    reason = claims_module.support(check(key=one_seed, direction="equivalent", margin=50_000),
                                   results)
    assert reason is not None and "beyond the margin" in reason
    assert claims_module.support(check(key="replay.net.vs_incumbent.flat",
                                       direction="equivalent", margin=500), results) is None


def test_interval_and_paired_case_claims() -> None:
    results = summary()
    interval = check(test="interval", key="llm.accuracy.vs_v1.v2")
    assert claims_module.support(interval, results) is None
    reason = claims_module.support(check(test="interval", key="llm.accuracy.vs_v1.v2",
                                         alpha=0.01), results)
    assert reason is not None and "below 1 - alpha" in reason  # a 95% interval at alpha 0.01
    assert "not a paired difference" in claims_module.support(
        check(test="interval", key="llm.consistency"), results)
    arms = check(test="mcnemar", key=None, keys=("replay.only_first", "replay.only_second"))
    assert claims_module.support(arms, results) is None
    reversed_arms = check(test="mcnemar", key=None,
                          keys=("replay.only_second", "replay.only_first"))
    assert "not positive" in claims_module.support(reversed_arms, results)
    means = check(test="mcnemar", key=None, keys=("replay.mean_first", "replay.mean_second"))
    assert "whole number of cases" in claims_module.support(means, results)


@pytest.mark.parametrize("keys", [
    ("replay.seed_mean_first", "replay.seed_mean_second"),  # means over seeds, whole numbers
    ("replay.mean_first", "replay.only_second"),  # 6.0 is a mean, not a count of cases
])
def test_mcnemar_needs_counts_of_cases_not_means(keys: tuple[str, str]) -> None:
    """Six against two on each of ten seeds is 60 against 20 cases, not six against two."""
    no_difference = check(test="mcnemar", key=None, keys=keys,
                          direction="no_detected_difference")
    assert "whole number of cases" in claims_module.support(no_difference, summary())


WORDING = claims_module.Wording.from_data({
    "policies": {"hybrid": "the hybrid policy", "incumbent": "the incumbent rules",
                 "flat": "the flat policy", "v1": "prompt v1", "v2": "prompt v2",
                 "v3": "prompt v3"},
    "metrics": {
        "net": {"more": "earned more net contribution than",
                "less": "earned less net contribution than", "noun": "net contribution",
                "format": "usd"},
        "accuracy": {"more": "was right more often than", "less": "was right less often than",
                     "noun": "accuracy", "format": "pp"},
        "right": {"more": "got more cases right than", "less": "got fewer cases right than",
                  "noun": "cases right", "format": "count"},
    },
    "families": {"baseline": "the baseline world", "surge": "the acquisition surge"},
    "capacities": {"base": "base review capacity", "high": "high review capacity"},
    "defaults": {"family": "baseline", "capacity": "base"},
})


def claim(**fields) -> claims_module.Claim:
    base = {"id": "hybrid-net", "policy": "hybrid", "reference": "incumbent", "metric": "net",
            "test": "sign", "direction": "positive", "key": "replay.net.vs_incumbent.hybrid"}
    return claims_module.Claim(**{**base, **fields})


def written(**fields) -> str:
    return claims_module.sentence(claim(**fields), summary(), WORDING)


def test_a_claim_is_written_from_its_record_and_the_results() -> None:
    assert written() == ("The hybrid policy earned more net contribution than the incumbent "
                         "rules on 9 of 10 seeds (exact sign test, p = 0.021).")
    assert written(key=None, family="surge", capacity="high") == (
        "In the acquisition surge at high review capacity, the hybrid policy earned more net "
        "contribution than the incumbent rules on 9 of 10 seeds (exact sign test, p = 0.021).")
    assert claim(key=None).derived_key(WORDING) == "evaluate.net.vs_incumbent.baseline.base.hybrid"


def test_each_test_and_direction_says_what_it_shows() -> None:
    flat = {"policy": "flat", "key": "replay.net.vs_incumbent.flat"}
    assert written(**flat, direction="no_detected_difference") == (
        "An exact sign test over 5 seeds detected no difference in net contribution between the "
        "flat policy and the incumbent rules (2 higher, 3 lower, p = 1.000).")
    assert written(**flat, direction="equivalent", margin=500) == (
        "On each of the 5 seeds, the net contribution of the flat policy was within $5 of that "
        "of the incumbent rules (largest difference $5).")
    accuracy = {"policy": "v2", "reference": "v1", "metric": "accuracy", "test": "interval",
                "key": "llm.accuracy.vs_v1.v2"}
    assert written(**accuracy) == (
        "Prompt v2 was right more often than prompt v1 on average: a mean difference of "
        "+12.5 pp (95% interval +4.0 pp to +21.0 pp).")
    unclear = {**accuracy, "policy": "v3", "key": "llm.accuracy.vs_v1.v3"}
    assert written(**unclear, direction="no_detected_difference") == (
        "The 95% interval of the mean difference in accuracy between prompt v3 and prompt v1, "
        "-3.0 pp to +4.0 pp, includes zero.")
    assert written(**unclear, direction="equivalent", margin=500) == (
        "The 95% interval of the mean difference in accuracy between prompt v3 and prompt v1, "
        "-3.0 pp to +4.0 pp, lies within ±5.0 pp.")
    pair = ("llm.right.vs_v1.v2.only_policy", "llm.right.vs_v1.v2.only_reference")
    assert written(policy="v2", reference="v1", metric="right", test="mcnemar", key=None,
                   keys=pair) == ("Prompt v2 got more cases right than prompt v1 (32 against 5 "
                                  "discordant cases, exact McNemar test, p < 0.001).")


def test_a_claim_the_results_no_longer_support_cannot_be_written() -> None:
    with pytest.raises(claims_module.ClaimError, match="the results do not support it: not "
                                                       "negative"):
        written(direction="negative")
    with pytest.raises(claims_module.ClaimError,
                       match="'replay.net.vs_incumbent.v1' is not in the summary"):
        written(policy="v1", key="replay.net.vs_incumbent.v1")
    sources = Sources(summary=summary(), claims={"hybrid-net": claim(direction="negative")},
                      wording=WORDING)
    with pytest.raises(RenderError) as error:
        render("Intro.\n\nResult: {{ claim:hybrid-net }}\n{{ claim:unknown }}\n", sources)
    assert error.value.problems[0].startswith("line 3: {{claim:hybrid-net}}: the results do "
                                              "not support it")
    assert "no claim 'unknown'" in error.value.problems[1]
    supported = Sources(summary=summary(), claims={"hybrid-net": claim()}, wording=WORDING)
    assert render("Result: {{ claim:hybrid-net }}", supported) == "Result: " + written()


@pytest.mark.parametrize(("fields", "message"), [
    ({"policy": "nobody"}, "its policy 'nobody' has no wording"),
    ({"reference": "hybrid"}, "compares a policy with itself"),
    ({"key": "replay.net.vs_incumbent.flat"}, "names policy ['flat'], but the claim says hybrid"),
    ({"key": "replay.net.vs_flat.hybrid"}, "names reference ['flat'], but the claim says "
                                            "incumbent"),
    ({"key": "replay.accuracy.vs_incumbent.hybrid"}, "names metric ['accuracy'], but the "
                                                       "claim says net"),
    ({"key": "replay.net.vs_incumbent.surge.hybrid"}, "names family ['surge'], but the claim "
                                                       "says baseline"),
    ({"key": None, "family": "surge", "capacity": "high", "policy": "flat"}, None),
    ({"test": "mcnemar", "key": None, "keys": ("x.right.vs_v1.v2.only_policy",
                                               "x.right.vs_v1.v3.only_reference")},
     "must be one paired comparison"),
    ({"test": "mcnemar", "keys": ("a.only_policy", "a.only_reference")},
     "the mcnemar test takes keys"),
    ({"direction": "equivalent"}, "a margin is required for equivalent"),
])
def test_a_claim_record_must_say_exactly_what_its_key_tests(fields: dict, message) -> None:
    if message is None:
        claim(**fields).check(WORDING)  # a derived key always matches its record
        return
    with pytest.raises(claims_module.ClaimError, match=re.escape(message)):
        claim(**fields).check(WORDING)


def test_the_wording_is_complete_and_unambiguous() -> None:
    data = {"policies": {"a": "A"}, "metrics": {}, "families": {"f": "F"},
            "capacities": {"c": "C"}, "defaults": {"family": "f", "capacity": "c"}}
    assert claims_module.Wording.from_data(data).policies == {"a": "A"}
    for broken, message in [
        ({"metrics": {"m": {"more": "x", "less": "y", "noun": "z"}}}, "needs more, less"),
        ({"metrics": {"m": {"more": "x", "less": "y", "noun": "z", "format": "nope"}}},
         "unknown format"),
        ({"defaults": {"family": "elsewhere", "capacity": "c"}}, "wording.defaults"),
        ({"families": {"a": "A", "f": "F"}}, "name two kinds of thing"),
        ({"policies": {"a": False}}, "give the words as text"),
    ]:
        with pytest.raises(ValueError, match=message):
            claims_module.Wording.from_data({**data, **broken})


def test_claims_parse_with_unique_ids_and_are_placed_by_templates() -> None:
    entry = {"id": "hybrid-net", "policy": "hybrid", "reference": "incumbent", "metric": "net",
             "test": "sign", "direction": "positive", "key": "replay.net.vs_incumbent.hybrid"}
    wording = {"policies": dict(WORDING.policies), "metrics": dict(WORDING.metrics),
               "families": dict(WORDING.families), "capacities": dict(WORDING.capacities),
               "defaults": {"family": "baseline", "capacity": "base"}}
    [parsed], _ = claims_module.parse_claims({"claims": [entry], "wording": wording})
    assert parsed == claim()
    with pytest.raises(ValueError, match="unique"):
        claims_module.parse_claims({"claims": [entry, entry], "wording": wording})
    with pytest.raises(ValueError, match="claim hybrid-net: its policy 'x' has no wording"):
        claims_module.parse_claims({"claims": [{**entry, "policy": "x"}], "wording": wording})
    assert claims_module.placed(["a {{ claim:hybrid-net }} b {{claim:other}}"]) == {
        "hybrid-net", "other"}
    assert claims_module.check_claims([claim()], summary(), WORDING, {"hybrid-net"}) == []
    assert claims_module.check_claims([claim(direction="negative")], summary(), WORDING,
                                      set()) == [
        "hybrid-net: no template places it ({{ claim:hybrid-net }})",
        "hybrid-net: the results do not support it: not negative at alpha 0.05: 9 positive, "
        "1 negative, p = 0.0215"]


def test_the_repository_wording_covers_what_the_evaluation_compares() -> None:
    import pipeline

    claims, wording = claims_module.load_claims()
    assert claims == []
    protocol = proto.load_protocol()
    assert set(protocol.raw["policies"]) <= set(wording.policies)
    assert set(pipeline.REFERENCES) <= set(wording.policies)
    assert set(pipeline.OUTCOME_METRICS) == set(wording.metrics)
    assert set(protocol.raw["families"]) <= set(wording.families)
    assert set(pipeline.expected_capacities(protocol)) <= set(wording.capacities)
    for name, words in wording.metrics.items():  # each difference prints in its own unit
        unit, _, denominator, _ = pipeline.OUTCOME_METRICS[name]
        formats.apply(words["format"], formats.Value(
            plain=1, unit=pipeline._difference_unit(unit), contrast=True), [])
        if denominator:  # a rate is worded as a rate, not as a count
            assert all("share" in words[side] or "rate" in words[side]
                       for side in ("more", "less")), name


# ---------------------------------------------------------------- the repository
def test_committed_documents_claims_and_templates_are_consistent() -> None:
    """Every committed document matches its template and the results, every claim
    still holds, and no template types a number by hand."""
    assert check_repository() == []


# ---------------------------------------------------------------- figures
def _figure():
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(3, 2))
    axes.bar(["rules", "hybrid"], [1.0, 2.0])
    axes.set_title("net contribution")
    return figure


def test_saved_svgs_carry_no_date_or_version_and_repeat_byte_for_byte(tmp_path: Path) -> None:
    from report import figures

    figure = _figure()
    first = figures.save_svg(figure, tmp_path / "a.svg").read_bytes()
    second = figures.save_svg(figure, tmp_path / "b.svg").read_bytes()
    assert first == second
    assert b"<dc:date>" not in first and b"Matplotlib v" not in first


def test_default_svgs_change_on_every_save_until_pinned(tmp_path: Path, monkeypatch) -> None:
    """A rebuild with identical data used to rewrite every committed SVG."""
    import matplotlib

    from report import figures

    monkeypatch.setitem(matplotlib.rcParams, "svg.hashsalt", None)
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    figure = _figure()
    figure.savefig(tmp_path / "a.svg", format="svg")
    figure.savefig(tmp_path / "b.svg", format="svg")
    assert (tmp_path / "a.svg").read_bytes() != (tmp_path / "b.svg").read_bytes()
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1")  # restored after the test; pin() overrides it
    figures.pin()
    figure.savefig(tmp_path / "c.svg", format="svg")
    figure.savefig(tmp_path / "d.svg", format="svg")
    assert (tmp_path / "c.svg").read_bytes() == (tmp_path / "d.svg").read_bytes()
