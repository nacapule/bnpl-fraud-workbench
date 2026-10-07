"""Rendered documents: formats, templates, the claims behind directional sentences, the lint."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.results import Interval, Metric, SeedSpread, StageResult, assemble_summary
from report import claims as claims_module
from report import formats, lint
from report.__main__ import check
from report.formats import FormatError, Value
from report.render import (
    RenderError,
    Sources,
    documents,
    out_of_sync,
    render,
    render_all,
)


def rate(numerator: int, denominator: int, **extra) -> Metric:
    return Metric.from_ratio(numerator, denominator, population="memos scored",
                             window="test", **extra)


def summary() -> dict:
    metrics = {
        "llm.accuracy.v1": rate(60, 200),
        "llm.accuracy.v2": rate(85, 200),
        "llm.consistency": rate(42, 50, interval=Interval(0.71, 0.92, "wilson")),
        "llm.accuracy_change": Metric(value=1250, unit="bps", population="v2 minus v1",
                                      window="test"),
        "replay.net.hybrid": Metric(
            value=123_456, unit="cents", population="net cash, test-window orders",
            window="test", seeds=SeedSpread({s: v for s, v in zip(range(1, 11), [
                150_000, 90_000, 200_000, -20_000, 130_000, 110_000, 95_000, 140_000,
                120_000, 219_560], strict=True)}),
        ),
        "replay.net.flat": Metric(
            value=0.0, unit="cents", population="net cash, test-window orders", window="test",
            seeds=SeedSpread({1: 500, 2: -400, 3: 300, 4: -100, 5: -300}),
        ),
        "replay.missing": Metric.not_evaluated(unit="rate", population="never-pay orders",
                                               window="test", reason="no orders",
                                               numerator=0, denominator=0),
        "replay.only_first": Metric(value=32, unit="count", population="cases", window="test"),
        "replay.only_second": Metric(value=5, unit="count", population="cases", window="test"),
    }
    stage = StageResult(
        stage="replay",
        versions={"world": "w1"},
        inputs={},
        metrics=metrics,
        tables={"replay.policies": [
            {"policy": "approve_all", "net_cents": 0, "held": 0.0, "evaluated": True},
            {"policy": "hybrid", "net_cents": 1_234_567, "held": 0.0123, "evaluated": False},
        ]},
    )
    return assemble_summary([stage])


@pytest.fixture
def sources() -> Sources:
    return Sources(
        summary=summary(),
        protocol={"windows": {"test": {"start": "2025-06-01"}}, "seeds": {"canonical": 416}},
        configs={"world": {"product": {"down_payment_bps": 2500}}},
    )


# ---------------------------------------------------------------- formats
def value_of(key: str) -> Value:
    from core.results import metric

    return Value(metric=metric(summary(), key))


def test_a_change_in_rates_is_percentage_points_not_percent(sources: Sources) -> None:
    """30.0% to 42.5% is +12.5 percentage points; printing it as "+12.5%" misstates it."""
    assert render("{{ llm.accuracy.v2 - llm.accuracy.v1 | pp }}", sources) == "+12.5 pp"
    assert render("{{ llm.accuracy.v2 - llm.accuracy.v1 }}", sources) == "+12.5 pp"
    assert render("{{ llm.accuracy_change | pp }}", sources) == "+12.5 pp"
    with pytest.raises(RenderError, match="percentage points"):
        render("{{ llm.accuracy.v2 - llm.accuracy.v1 | pct }}", sources)


def test_rates_show_their_own_numerator_and_denominator(sources: Sources) -> None:
    text = render("{{ llm.consistency }} ({{ llm.consistency | n }}, {{ llm.consistency | ci }})",
                  sources)
    assert text == "84.0% (42/50, 95% CI 71.0% to 92.0%)"


def test_money_counts_and_signs() -> None:
    cents = Value(metric=Metric(value=-123_456, unit="cents", population="p", window="test"))
    assert formats.apply("usd", cents, []) == "-$1,235"
    assert formats.apply("usd", cents, ["2"]) == "-$1,234.56"
    small = Value(metric=Metric(value=-4, unit="cents", population="p", window="test"))
    assert formats.apply("usd", small, ["signed"]) == "$0"  # no sign on a value that rounds to 0
    gain = Value(metric=Metric(value=50_000, unit="cents", population="p", window="test"))
    assert formats.apply("usd", gain, ["signed"]) == "+$500"
    count = Value(metric=Metric(value=12_345, unit="count", population="p", window="test"))
    assert formats.apply("value", count, []) == "12,345"
    held = Value(metric=Metric(value=37.25, unit="bps", population="p", window="test"))
    assert formats.apply("per_10k", held, []) == "37.2 per 10,000"
    assert formats.apply("bps", held, ["2"]) == "37.25 bps"


def test_formats_refuse_the_wrong_unit() -> None:
    with pytest.raises(FormatError, match="does not fit unit"):
        formats.apply("usd", value_of("llm.consistency"), [])
    with pytest.raises(FormatError, match="does not fit unit"):
        formats.apply("pct", value_of("replay.net.hybrid"), [])
    with pytest.raises(FormatError, match="whole number"):
        formats.apply("count", Value(metric=Metric(value=1.5, unit="count", population="p",
                                                   window="test")), [])
    with pytest.raises(FormatError, match="unknown format"):
        formats.apply("percent", value_of("llm.consistency"), [])
    with pytest.raises(FormatError, match="unknown format argument"):
        formats.apply("pct", value_of("llm.consistency"), ["wide"])


def test_unevaluated_metrics_say_so_with_their_n(sources: Sources) -> None:
    assert render("{{ replay.missing }}", sources) == "not evaluated (N=0)"
    assert render("{{ replay.missing | pct }}", sources) == "not evaluated (N=0)"
    assert render("{{ replay.missing | define }}", sources) == "never-pay orders (test window)"


def test_seed_replication_is_reported_with_mean_range_and_sign_count(sources: Sources) -> None:
    text = render("{{ replay.net.hybrid | seeds }} ({{ replay.net.hybrid | p }})", sources)
    assert text == ("mean +$1,235, min -$200, max +$2,196; positive on 9/10 seeds (p = 0.021)")
    assert render("{{ replay.net.flat | signs }}", sources) == "negative on 3/5 seeds"
    tie = Value(metric=Metric(value=0.0, unit="cents", population="p", window="test",
                              seeds=SeedSpread({1: 5, 2: -5})))
    assert formats.apply("signs", tie, []) == "mixed: 1 positive and 1 negative of 2 seeds"


def test_fixed_definitions_from_the_protocol_and_configuration(sources: Sources) -> None:
    assert render("{{ protocol:windows.test.start | date }}", sources) == "1 June 2025"
    assert render("{{ protocol:seeds.canonical }}", sources) == "416"
    assert render("{{ config:world:product.down_payment_bps | bps_pct }}", sources) == "25%"


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


def test_generated_tables_name_their_columns(sources: Sources) -> None:
    template = ('Before\n{{ table:replay.policies | policy "Policy", net_cents "Net" usd, '
                'held "Held" pct:2, evaluated }}\nAfter')
    assert render(template, sources).split("\n") == [
        "Before",
        "| Policy | Net | Held | evaluated |",
        "|---|---:|---:|---|",
        "| approve_all | $0 | 0.00% | yes |",
        "| hybrid | $12,346 | 1.23% | no |",
        "After",
    ]
    with pytest.raises(RenderError, match="own line"):
        render("See {{ table:replay.policies | policy }} here", sources)
    with pytest.raises(RenderError, match="no column"):
        render("{{ table:replay.policies | policy, missing }}", sources)


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


# ---------------------------------------------------------------- lint
def test_numbers_typed_into_a_template_are_flagged() -> None:
    template = (
        "The hybrid policy saved 12% of loss across 10 seeds.\n"
        "Rendered: {{ replay.net.hybrid | usd }} and {{ replay.net.hybrid | seeds }}.\n"
    )
    findings = lint.number_findings("README.md", template)
    assert [(finding.line, finding.text) for finding in findings] == [(1, "12"), (1, "10")]


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


def test_directional_sentences_need_a_claim() -> None:
    rendered = (
        "# Results\n\n"
        "The hybrid policy earned more than the incumbent rules on 9/10 seeds. "
        "Review capacity was the same for every policy.\n\n"
        "| policy | better |\n|---|---|\n| hybrid | yes |\n\n"
        "- Lower friction for new customers came at no cost.\n"
    )
    findings = lint.directional_findings("README.md", rendered, [], ["more", "lower", "better"])
    assert [(f.line, f.text) for f in findings] == [
        (3, "The hybrid policy earned more than the incumbent rules on 9/10 seeds."),
        (9, "- Lower friction for new customers came at no cost."),
    ]
    covered = lint.directional_findings(
        "README.md", rendered, ["earned more than the incumbent rules"], ["more", "lower"],
        ["- Lower friction for new customers came at no cost."],
    )
    assert covered == []


def test_published_documents_need_templates(tmp_path: Path) -> None:
    root, templates = _repo(tmp_path, "text\n")
    (root / "cases").mkdir()
    (root / "cases" / "CASE-01.md").write_text("rendered\n")
    (root / "cases" / "CASE-02.md").write_text("by hand\n")
    (root / "README.md").write_text("by hand\n")
    config = {"documents": ["README.md", "cases/*.md"], "not_yet_templated": ["README.md",
                                                                               "cases/OLD.md"]}
    assert lint.missing_templates(config, root, templates) == [
        "cases/CASE-02.md: published document without a template",
        "cases/OLD.md: listed in not_yet_templated but does not exist",
    ]


# ---------------------------------------------------------------- claims
def claim(**fields) -> claims_module.Claim:
    base = {"id": "hybrid-earns-more", "document": "README.md",
            "text": "The hybrid policy earned more than the incumbent rules",
            "test": "sign", "direction": "positive", "key": "replay.net.hybrid"}
    base.update(fields)
    return claims_module.Claim(**base)


def test_a_supported_claim_holds_and_a_reversed_one_fails() -> None:
    results = summary()
    assert claims_module.support(claim(), results) is None
    assert "not negative" in claims_module.support(claim(direction="negative"), results)
    assert "significant" in claims_module.support(claim(direction="no_difference"), results)
    flat = claim(key="replay.net.flat")
    assert "not positive" in claims_module.support(flat, results)
    assert claims_module.support(claim(key="replay.net.flat", direction="no_difference"),
                                 results) is None


def test_a_majority_of_seeds_is_not_enough_without_the_test() -> None:
    """Four of five seeds positive is p = 0.375: not a directional result at 0.05."""
    item = Metric(value=1.0, unit="cents", population="p", window="test",
                  seeds=SeedSpread({1: 5, 2: 3, 3: 2, 4: 1, 5: -1}))
    stage = StageResult(stage="replay", versions={}, inputs={}, metrics={"replay.x": item})
    reason = claims_module.support(claim(key="replay.x"), assemble_summary([stage]))
    assert reason is not None and "p = 0.3750" in reason


def test_interval_and_paired_case_claims() -> None:
    results = summary()
    consistency = claim(test="interval", key="llm.consistency")
    assert claims_module.support(consistency, results) is None
    arms = claim(test="mcnemar", key=None, keys=("replay.only_first", "replay.only_second"))
    assert claims_module.support(arms, results) is None
    reversed_arms = claim(test="mcnemar", key=None,
                          keys=("replay.only_second", "replay.only_first"))
    assert "not positive" in claims_module.support(reversed_arms, results)


def test_claims_fail_when_their_sentence_leaves_the_document() -> None:
    results = summary()
    document = "Overall, the hybrid policy\nearned more than the incumbent rules."
    assert claims_module.check_claims([claim(text="the hybrid policy earned more than the "
                                             "incumbent rules")], results, lambda _: document) == []
    problems = claims_module.check_claims([claim()], results, lambda _: "Something else.")
    assert problems == ["hybrid-earns-more: its text is not in README.md"]
    missing = claims_module.check_claims([claim(key="replay.gone")], results, lambda _: None)
    assert missing[0] == "hybrid-earns-more: document README.md does not exist"
    assert "replay.gone" in missing[1]


@pytest.mark.parametrize(
    "fields",
    [{"test": "t"}, {"direction": "up"}, {"text": " "}, {"test": "mcnemar"},
     {"keys": ("a", "b")}, {"alpha": 1.5}],
)
def test_malformed_claims_are_rejected(fields: dict) -> None:
    with pytest.raises(ValueError):
        claim(**fields)


def test_claim_ids_are_unique() -> None:
    entry = {"id": "a", "document": "README.md", "text": "x", "test": "sign",
             "direction": "positive", "key": "k"}
    with pytest.raises(ValueError, match="unique"):
        claims_module.parse_claims({"claims": [entry, entry]})


# ---------------------------------------------------------------- the repository
def test_committed_documents_claims_and_templates_are_consistent() -> None:
    """Every committed document matches its template and the results, every claim
    still holds, and no template types a number by hand."""
    assert check() == []


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
