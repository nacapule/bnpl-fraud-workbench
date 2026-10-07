"""Rendered documents: formats, templates, the claims behind directional sentences, the lint."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def test_directional_sentences_need_a_claim_whose_clauses_cover_them() -> None:
    rendered = (
        "# Results\n\n"
        "The hybrid policy earned more than the incumbent rules on 9/10 seeds. "
        "Review capacity was the same for every policy.\n\n"
        "| policy | better |\n|---|---|\n| hybrid | yes |\n\n"
        "- Lower friction for new customers came at no cost.\n"
        "<!-- Hybrid did better everywhere. -->\n"
    )
    words = ["more", "lower", "better"]
    findings = lint.directional_findings("README.md", rendered, {}, words)
    assert [(f.line, f.text) for f in findings] == [
        (3, "The hybrid policy earned more than the incumbent rules on 9/10 seeds."),
        (9, "Lower friction for new customers came at no cost."),
    ]
    sentence = "The hybrid policy earned more than the incumbent rules on 9/10 seeds."
    claimed = {sentence: ["earned more than the incumbent rules"]}
    allowed = ["Lower friction for new customers came at no cost."]
    assert lint.directional_findings("README.md", rendered, claimed, words, allowed) == []
    two = "Hybrid earned more than rules and had lower loss than rules."
    [finding] = lint.directional_findings("README.md", two, {two: ["earned more than rules"]},
                                          words)
    assert finding.text.endswith("(no check covers: lower)")
    assert lint.directional_findings(
        "README.md", two, {two: ["earned more than rules", "had lower loss than rules"]},
        words) == []


def test_code_blocks_and_headings_do_not_join_sentences() -> None:
    text = (
        "# Results\n"
        "Hybrid earned more than rules, i.e. by a margin.\n\n"
        "    Hybrid earned more than everything.\n\n"
        "- First item. Second sentence.\n"
        "    continued here.\n"
        "1. Numbered item\n"
    )
    assert lint.sentences(text) == [
        (1, "Results"),
        (2, "Hybrid earned more than rules, i.e. by a margin."),
        (6, "First item."),
        (6, "Second sentence."),
        (7, "continued here."),
        (8, "Numbered item"),
    ]


@pytest.mark.parametrize(("text", "expected"), [
    (">     Hybrid earned more than incumbent rules.\n", []),  # code in a quote
    ("- Item.\n\n        Hybrid earned more than incumbent rules.\n", ["Item."]),  # in a list
    ("- Item.\n  - Nested.\n\n          Hybrid earned more.\n", ["Item.", "Nested."]),
    ("- Item.\n\n  ```\n  Hybrid earned more.\n  ```\n", ["Item."]),  # fenced, in a list
    ("> ```\n> Hybrid earned more.\n> ```\n", []),  # fenced, in a quote
    ("# Results\n    Hybrid earned more.\n", ["Results"]),  # code after a heading
    ("- Item.\n\n  Hybrid earned more than incumbent rules.\n",  # a list paragraph
     ["Item.", "Hybrid earned more than incumbent rules."]),
    ("> Hybrid earned more\n> than incumbent rules.\n",  # one quoted paragraph
     ["Hybrid earned more than incumbent rules."]),
    ("> Hybrid earned more\nthan incumbent rules.\n",  # continued without the mark
     ["Hybrid earned more than incumbent rules."]),
    ("**Claim.** Next sentence.\n", ["Claim.", "Next sentence."]),
    ("Hybrid [earned more](x.md) than `rules`.\n", ["Hybrid earned more than rules."]),
    ("* Hybrid earned more\n* than rules\n", ["Hybrid earned more", "than rules"]),
    ("Results\n=======\nHybrid earned more.\n", ["Results", "Hybrid earned more."]),
    ("The net_contribution_cents key.\n", ["The net_contribution_cents key."]),
])
def test_sentences_are_what_markdown_shows(text: str, expected: list[str]) -> None:
    """Code inside quotes and lists is hidden; quotes, emphasis and links read as text."""
    assert [sentence for _, sentence in lint.sentences(text)] == expected


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
    base = {"clause": "earned more than the incumbent rules", "test": "sign",
            "direction": "positive", "key": "replay.net.vs_incumbent.hybrid"}
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


def claim(sentence: str = "The hybrid policy earned more than the incumbent rules.",
          **checks) -> claims_module.Claim:
    return claims_module.Claim(id="hybrid-earns-more", document="README.md", sentence=sentence,
                               checks=(check(**checks),))


VOCABULARY = claims_module.Vocabulary.from_config({
    "names": {
        "policies": {"hybrid": ["hybrid"], "incumbent": ["incumbent rules"],
                     "approve_all": ["approve-all"], "boosting": ["gradient boosting"],
                     "expected_loss": ["expected loss"]},
        "metrics": {"net": ["earned", "net contribution"], "loss": ["loss", "lost"]},
        "families": {"surge": ["acquisition surge"]},
    },
    "references": ["incumbent", "approve_all"],
    "directional_words": ["more", "less", "lower", "higher"],
})


def problems_of(sentence: str, *checks: dict, context: tuple[str, ...] = ()) -> list[str]:
    claim_ = claims_module.Claim(id="c", document="README.md", sentence=sentence,
                                 checks=tuple(check(**fields) for fields in checks),
                                 context=context)
    return claims_module.name_problems(claim_, VOCABULARY)


def test_every_policy_named_in_a_claim_needs_a_check_against_it() -> None:
    """"More than the incumbent rules and approve-all" makes two comparisons."""
    results = summary()
    sentence = "The hybrid policy earned more than the incumbent rules and approve-all."
    document = sentence
    one = claims_module.Claim(id="c", document="README.md", sentence=sentence,
                              checks=(check(clause="The hybrid policy earned more than the "
                                            "incumbent rules"),))
    problems = claims_module.check_claims([one], results, lambda _: document, VOCABULARY)
    assert problems == ["c: 'approve-all' (approve_all) is in no check's clause that tests it"]
    wrong = claims_module.Claim(id="c", document="README.md", sentence=sentence,
                                checks=(check(clause=sentence[:-1]),))
    assert "c: check 1: its clause names approve_all, which its key does not test" in \
        claims_module.check_claims([wrong], results, lambda _: document, VOCABULARY)
    off = claims_module.Claim(id="c", document="README.md", sentence=sentence,
                              checks=(check(clause="earned less"),))
    assert "c: check 1: its clause is not in the sentence" in claims_module.check_claims(
        [off], results, lambda _: document, VOCABULARY)


def test_each_comparison_in_a_sentence_needs_its_own_check() -> None:
    """An earnings result cannot stand behind a claim about loss in the same sentence."""
    sentence = "Hybrid earned more than incumbent rules and had lower loss than incumbent rules."
    whole = {"clause": sentence[:-1]}
    assert problems_of(sentence, whole) == [
        "check 1: its clause makes 2 comparisons (more, lower); give each its own check",
        "check 1: its clause names loss, which its key does not test",
        "'loss' (loss) is in no check's clause that tests it",
    ]
    earned = {"clause": "Hybrid earned more than incumbent rules"}
    loss = {"clause": "had lower loss than incumbent rules", "direction": "negative",
            "key": "replay.loss.vs_incumbent.hybrid"}
    assert problems_of(sentence, earned, loss) == []
    assert problems_of(sentence, earned, {**loss, "key": "replay.net.vs_incumbent.hybrid"}) == [
        "check 2: its clause names loss, which its key does not test",
        "'loss' (loss) is in no check's clause that tests it",
    ]


def test_a_check_tests_only_what_the_sentence_names() -> None:
    """The policy, reference and metric of each check's key must be in the sentence."""
    sentence = "Hybrid earned more and lost less than incumbent rules."
    earned = {"clause": "Hybrid earned more", "key": "replay.net.vs_approve_all.hybrid"}
    lost = {"clause": "lost less than incumbent rules", "direction": "negative",
            "key": "replay.loss.vs_incumbent.hybrid"}
    assert problems_of(sentence, earned, lost) == [
        "check 1: its key tests approve_all, which the sentence does not name"]
    assert problems_of(sentence, {**earned, "key": "replay.net.vs_incumbent.hybrid"}, lost) == []
    other = {**lost, "key": "replay.loss.vs_incumbent.boosting"}
    assert problems_of(sentence, {**earned, "key": "replay.net.vs_incumbent.hybrid"}, other) == [
        "check 2: its key tests boosting, which the sentence does not name"]


def test_a_policy_mentioned_in_passing_is_declared_as_context() -> None:
    sentence = ("Hybrid, which ranks the queue with gradient boosting, earned more than "
                "incumbent rules.")
    clause = {"clause": sentence[:-1]}
    aside = "which ranks the queue with gradient boosting"
    assert problems_of(sentence, clause) == [
        "check 1: its clause names boosting, which its key does not test",
        "'gradient boosting' (boosting) is in no check's clause that tests it",
    ]
    assert problems_of(sentence, clause, context=(aside,)) == []
    # An aside names no comparison and no reference, and is set off from the sentence.
    compared = "Hybrid earned more than incumbent rules and gradient boosting."
    assert problems_of(compared, {"clause": compared[:-1]}, context=("and gradient boosting",)) \
        [0].startswith("context 'and gradient boosting' is not an aside")
    assert problems_of(compared, {"clause": compared[:-1]}, context=("gradient boosting",)) \
        [0].startswith("context 'gradient boosting' is not an aside")
    loose = "Hybrid, earning more than gradient boosting, earned more than incumbent rules."
    assert "context 'earning more than gradient boosting' makes a comparison (more)" in \
        problems_of(loose, {"clause": loose[:-1]},
                    context=("earning more than gradient boosting",))
    reference = "Hybrid, unlike incumbent rules, earned more than approve-all."
    assert "context 'unlike incumbent rules' names incumbent, which comparisons are against" in \
        problems_of(reference, {"clause": reference[:-1],
                                "key": "replay.net.vs_approve_all.hybrid"},
                    context=("unlike incumbent rules",))
    assert problems_of(sentence, clause, context=("not here",)) == [
        "context 'not here' is not in the sentence",
        "check 1: its clause names boosting, which its key does not test",
        "'gradient boosting' (boosting) is in no check's clause that tests it",
    ]


def test_names_match_the_longest_phrase() -> None:
    """"Expected loss" is a policy; "lost" alone is the loss metric."""
    text = "The expected loss policy lost less in the acquisition surge."
    assert [name for _, name in claims_module.mentions(VOCABULARY.names, text)] == [
        "expected_loss", "loss", "surge"]


def test_the_vocabulary_is_unambiguous() -> None:
    with pytest.raises(ValueError, match="names both"):
        claims_module.Vocabulary.from_config(
            {"names": {"policies": {"a": ["x"]}, "metrics": {"b": ["X"]}}})
    with pytest.raises(ValueError, match="listed twice"):
        claims_module.Vocabulary.from_config(
            {"names": {"policies": {"a": ["x"]}, "metrics": {"a": ["y"]}}})
    with pytest.raises(ValueError, match="references must be policies"):
        claims_module.Vocabulary.from_config(
            {"names": {"metrics": {"a": ["y"]}}, "references": ["a"]})


def test_the_documented_claim_example_passes_the_repository_vocabulary() -> None:
    text = (REPO / "report" / "claims.yaml").read_text()
    example = "\n".join(line[2:] for line in text.split("\n") if line.startswith("# ")
                        and not line.startswith("# Every") and "#   #" not in line)
    example = example[example.index("- id:"):]
    [parsed] = claims_module.parse_claims({"claims": yaml.safe_load(example)})
    vocabulary = claims_module.Vocabulary.from_config(lint.load_config())
    assert claims_module.name_problems(parsed, vocabulary) == []


def test_claims_fail_when_their_sentence_leaves_the_visible_document() -> None:
    results = summary()
    document = "Overall:\n\nThe hybrid policy\nearned more than the incumbent rules. Next."
    assert claims_module.check_claims([claim()], results, lambda _: document) == []
    hidden = "<!-- The hybrid policy earned more than the incumbent rules. -->\nOther text."
    assert claims_module.check_claims([claim()], results, lambda _: hidden) == [
        "hybrid-earns-more: its sentence is not a visible sentence of README.md"
    ]
    longer = "The hybrid policy earned more than the incumbent rules and cut loss."
    assert claims_module.check_claims([claim()], results, lambda _: longer) != []
    missing = claims_module.check_claims([claim(key="replay.vs_x.gone")], results,
                                         lambda _: None)
    assert missing[0] == "hybrid-earns-more: document README.md does not exist"
    assert "replay.vs_x.gone" in missing[1]


@pytest.mark.parametrize(
    "fields",
    [{"clause": " "}, {"test": "t"}, {"direction": "up"}, {"test": "mcnemar"},
     {"keys": ("a", "b")},
     {"alpha": 1.5}, {"direction": "equivalent"}, {"margin": 5.0},
     {"test": "mcnemar", "key": None, "keys": ("a", "b"), "direction": "equivalent",
      "margin": 1.0}],
)
def test_malformed_checks_are_rejected(fields: dict) -> None:
    with pytest.raises(ValueError):
        check(**fields)


def test_claims_parse_and_ids_are_unique() -> None:
    entry = {"id": "a", "document": "README.md", "sentence": "x.",
             "checks": [{"clause": "x", "test": "sign", "direction": "positive",
                         "key": "r.vs_a.k"}]}
    [parsed] = claims_module.parse_claims({"claims": [entry]})
    assert parsed.checks[0].key == "r.vs_a.k"
    with pytest.raises(ValueError, match="unique"):
        claims_module.parse_claims({"claims": [entry, entry]})
    with pytest.raises(ValueError, match="at least one check"):
        claims_module.parse_claims({"claims": [{**entry, "checks": []}]})


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
