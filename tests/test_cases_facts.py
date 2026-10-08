"""Case facts: the evidence is the row the replay decided on and nothing known after it,
the recorded action is the replay's, later outcomes and latent truth are kept apart, and
every amount reconciles with the ledger.

The world is the hand-checked mini world (``tests/fixtures/mini_world/README.md`` gives
each order's net cash), replayed by the incumbent rules at two operating points: one that
auto-declines the takeover, the never-pay order and nothing else of interest, and one
that sends them to review. The oracles are the README's cash, the world's own rows and
hand arithmetic, never the facts code.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cases import facts as facts_module
from cases import select as select_module
from cases.rule import RuleError
from core import asof, config, world
from model.train import RuleScorer
from queue_sim import policies, stage
from rules import definitions, engine

MINI = Path(__file__).resolve().parent / "fixtures" / "mini_world"
WINDOW = (pd.Timestamp("2024-12-01"), pd.Timestamp("2025-07-01"))
OBSERVED = pd.Timestamp("2025-06-30 23:59:59")
# README "Net cash (cents)", approve-all: dan's takeover, eve's stolen card, nr3's never-pay
README_NET = {12: -49_445, 18: -53_500, 22: -22_550}
FILES = {"account_takeover", "never_pay_vs_hardship", "traveller", "card_testing", "ring"}


class _Calibrator:
    def predict(self, scores):
        return np.zeros(len(scores))


def _incumbent(review: float, decline: float) -> policies.Policy:
    scorer = RuleScorer("rules", "mini", definitions.COLUMNS, engine.score, _Calibrator())
    return policies.single_scorer("incumbent_rules", scorer).with_thresholds(review, decline)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return world.read_world(MINI)


def make_run(tmp: Path, tables, review: float, decline: float) -> Path:
    """A pipeline run directory holding the mini world as the canonical world, with what
    the replay keeps from the incumbent's run and the tune stage's chosen thresholds."""
    policy = _incumbent(review, decline)
    bench = stage.Bench.of(tables, asof.build_context(tables), seed=416, observed_until=OBSERVED)
    result = bench.run(policy, WINDOW, stage.base_staffing(bench.policy_cfg))
    frames = stage.incumbent_frames(policy, result)
    world_dir = tmp / "worlds" / "416-baseline"
    shutil.copytree(MINI, world_dir)
    for name, file_name in (("decisions", "review_decisions.pkl"),
                            ("checkout_rows", "checkout_rows.pkl"),
                            ("fates", "incumbent_fates.pkl")):
        frames[name].to_pickle(world_dir / file_name)
    (tmp / "fit" / "416").mkdir(parents=True)
    (tmp / "fit" / "416" / "model_rules.json").write_text(json.dumps({
        "version": "mini", "columns": list(definitions.COLUMNS),
        "calibration": {"thresholds": [0.0, 100.0], "probabilities": [0.0, 1.0]}}))
    (tmp / "results").mkdir()
    (tmp / "results" / "tune.json").write_text(json.dumps({"tables": {"tune.chosen": [
        {"seed": 416, "policy": "incumbent_rules", "chosen_version": policy.version,
         "review_threshold": review, "decline_threshold": decline},
        {"seed": 416, "policy": "logistic", "chosen_version": "other",
         "review_threshold": 0.5, "decline_threshold": None}]}}))
    return tmp


@pytest.fixture(scope="module")
def declines(tables, tmp_path_factory) -> Path:
    return make_run(tmp_path_factory.mktemp("declines"), tables, 35.0, 40.0)


@pytest.fixture(scope="module")
def reviews(tables, tmp_path_factory) -> Path:
    return make_run(tmp_path_factory.mktemp("reviews"), tables, 35.0, 50.0)


@pytest.fixture(scope="module")
def declined_facts(declines):
    return facts_module.build(facts_module.load_run(declines))


@pytest.fixture(scope="module")
def reviewed_facts(reviews):
    return facts_module.build(facts_module.load_run(reviews))


def alert(facts, file, slot=None):
    entry = facts[file]
    return entry["alerts"][slot or entry["primary_alert"]]


# ------------------------------------------------------------------ selection and files


def test_every_file_is_written_and_missing_slots_carry_their_counts(declined_facts):
    assert set(declined_facts) == FILES
    assert alert(declined_facts, "account_takeover")["publication"]["order_id"] == 12
    assert alert(declined_facts, "never_pay_vs_hardship")["publication"]["order_id"] == 22
    assert alert(declined_facts, "card_testing")["publication"]["order_id"] == 18
    for file, slot in (("never_pay_vs_hardship", "hardship"), ("traveller", None), ("ring", None)):
        missing = alert(declined_facts, file, slot)
        assert missing["selected"] is False
        assert missing["publication"]["order_id"] is None
        assert missing["publication"]["candidates"]["eligible"] == 0
        assert set(missing) == {"selected", "slot", "publication"}


def test_publication_names_the_alert_and_its_selection(declined_facts, tables):
    import hashlib

    pub = alert(declined_facts, "account_takeover")["publication"]
    version = _incumbent(35.0, 40.0).version
    assert pub["alert_id"] == f"12:{version}" and pub["policy_version"] == version
    assert pub["selection_hash"] == hashlib.sha256(b"bnpl-cases-2026-10:12").hexdigest()
    assert pub["tier"] == "checkout_alert"  # the takeover was declined at checkout
    assert pub["candidates"] == {"eligible": 1, "reviewed": 0, "available": 1,
                                 "tiers": {"reviewed_alert": 0, "checkout_alert": 1}}


# ------------------------------------------------------------------ evidence at decision time


def _hand_history(tables, user: int, at: pd.Timestamp) -> dict[str, int]:
    """Approve-all history of an account before ``at``, counted by hand from the world."""
    attempts = tables["order_attempts"]
    earlier = attempts.loc[(attempts["user_id"] == user) & (attempts["known_at"] < at)
                           & (attempts["processor_result"] == "approved"), "order_id"]
    plans = tables["plans"].loc[tables["plans"]["order_id"].isin(earlier), "plan_id"]
    schedule = tables["installment_schedule"]
    due = schedule.loc[schedule["plan_id"].isin(plans) & (schedule["seq"] >= 1)
                       & (schedule["due_at"] < at)]
    payments = tables["payment_attempts"]
    paid = payments.loc[(payments["result"] == "success") & (payments["known_at"] < at)]
    paid_keys = set(zip(paid["plan_id"], paid["seq"], strict=True))
    reports = tables["victim_reports"]
    return {"approved_orders_user_ever": len(earlier), "installments_due_user": len(due),
            "installments_paid_user": sum((p, s) in paid_keys
                                          for p, s in zip(due["plan_id"], due["seq"], strict=True)),
            "victim_reports_user": int(((reports["user_id"] == user)
                                        & (reports["known_at"] < at)).sum())}


@pytest.mark.parametrize("which", ["declined_facts", "reviewed_facts"])
def test_the_evidence_is_the_saved_row_and_knows_nothing_later(which, request, tables):
    built = request.getfixturevalue(which)
    for file in ("account_takeover", "never_pay_vs_hardship", "card_testing"):
        facts = alert(built, file)
        decision, row = facts["decision"], facts["evidence"]["row"]
        at = pd.Timestamp(decision["evidence_at"])
        assert row["decision_at"] == decision["evidence_at"]
        assert pd.Timestamp(decision["checkout_at"]) <= at <= pd.Timestamp(decision["action_at"])
        assert set(row) == set(asof.KEY_COLUMNS) | set(asof.COLUMN_NAMES)
        hand = _hand_history(tables, row["user_id"], at)
        assert {key: row[key] for key in hand} == hand
        # nothing from the later block leaks in: the ledger and the label are known later
        assert "label" not in json.dumps(facts["evidence"])
        assert pd.Timestamp(facts["later"]["label"]["label_known_at"]) > at


def test_a_takeover_reported_later_is_not_evidence_at_the_decision(declined_facts, tables):
    facts = alert(declined_facts, "account_takeover")
    reports = tables["victim_reports"]
    assert ((reports["user_id"] == 1) & (reports["order_id"] == 12)).sum() == 1  # reported later
    assert facts["evidence"]["row"]["victim_reports_user"] == 0
    assert "victim report" not in [e["event"] for e in facts["later"]["incumbent"]["events"]]
    assert "victim report" in [e["event"] for e in facts["later"]["approve_all"]["events"]]


def test_reviewed_alerts_rest_on_the_first_decision_and_report_later_checks_apart(
        reviewed_facts, reviews):
    kept = pd.read_pickle(reviews / "worlds" / "416-baseline" / "review_decisions.pkl")
    facts = alert(reviewed_facts, "account_takeover")
    saved = kept.loc[kept["order_id"] == 12].iloc[0]
    assert facts["publication"]["tier"] == "reviewed_alert"
    assert facts["decision"]["kind"] == "review"
    assert facts["decision"]["evidence_at"] == str(saved["decision_at"])
    assert facts["decision"]["analyst_started_at"] == str(saved["taken_up_at"])
    assert facts["decision"]["action_at"] == str(saved["decided_at"])
    assert facts["decision"]["recorded_action"] == saved["disposition"] == "hold"
    assert facts["reviewer"]["standard_disposition"] == "hold"
    for column in asof.COLUMN_NAMES:
        value = saved[column]
        expected = None if pd.isna(value) else round(float(value), 4)
        assert facts["evidence"]["row"][column] == expected, column
    later = facts["later"]["review"]
    assert later["first_disposition"] == "hold" and later["final"] == saved["final"]
    assert [c["check"] for c in later["checks"]] == [str(c) for c, _, _ in saved["checks_later"]]
    assert all(pd.Timestamp(c["completed_at"]) > pd.Timestamp(facts["decision"]["action_at"])
               for c in later["checks"])


def test_a_checkout_alert_that_was_also_reviewed_keeps_the_checkout_row(reviewed_facts, reviews):
    kept = pd.read_pickle(reviews / "worlds" / "416-baseline" / "review_decisions.pkl")
    facts = alert(reviewed_facts, "card_testing")
    assert facts["publication"]["tier"] == "checkout_alert"
    assert facts["decision"]["kind"] == "checkout"
    assert facts["decision"]["evidence_at"] == facts["decision"]["checkout_at"]
    assert facts["decision"]["recorded_action"] == "review"
    assert facts["evidence"]["row"]["processor_declines_device_24h"] == 2
    assert facts["later"]["review"]["final"] == kept.set_index("order_id").loc[18, "final"]
    assert "reviewer" not in facts


def test_same_day_orders_are_the_decisions_the_days_evidence_does_not_reflect():
    fates = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5], "user_id": [7, 7, 7, 7, 8],
        "checkout_at": pd.to_datetime(["2025-06-27 23:50", "2025-06-28 09:26", "2025-06-28 09:33",
                                       "2025-06-28 09:58", "2025-06-28 09:30"]),
        "route": ["approve", "approve", "auto_decline", "auto_decline", "review"]})
    assert facts_module.same_day_orders(fates, 4, "2025-06-28 09:58:00") == [
        {"order_id": 2, "checkout_at": pd.Timestamp("2025-06-28 09:26"), "route": "approve"},
        {"order_id": 3, "checkout_at": pd.Timestamp("2025-06-28 09:33"), "route": "auto_decline"}]
    # a review's evidence assembled at the start of a day reflects everything before it
    assert facts_module.same_day_orders(fates, 4, "2025-06-28 00:00:00") == []


def test_the_facts_list_the_same_day_orders(declined_facts):
    for file in ("account_takeover", "never_pay_vs_hardship", "card_testing"):
        assert alert(declined_facts, file)["decision"]["same_day_orders"] == []


# ------------------------------------------------------------------ amounts and the ledger


def _terms() -> dict:
    return config.load("world")["product"]


@pytest.mark.parametrize("which", ["declined_facts", "reviewed_facts"])
def test_amounts_reconcile_with_the_ledger(which, request, tables):
    built = request.getfixturevalue(which)
    terms = _terms()
    cash_events = tables["cash_events"]
    for file in ("account_takeover", "never_pay_vs_hardship", "card_testing"):
        facts = alert(built, file)
        order = facts["publication"]["order_id"]
        later = facts["later"]
        # approve-all: the world's cash for the order, row by row, and the README's net
        rows = cash_events.loc[cash_events["order_id"] == order]
        world_view = later["approve_all"]
        assert world_view["net_cents"] == README_NET[order] == int(rows["amount_cents"].sum())
        assert sorted((c["event_id"], c["amount_cents"]) for c in world_view["cash"]) == sorted(
            zip(rows["event_id"].tolist(), rows["amount_cents"].tolist(), strict=True))
        for view in (later["incumbent"], world_view):
            assert view["net_cents"] == sum(c["amount_cents"] for c in view["cash"])
            assert view["net_cents"] == sum(view["cash_by_kind"].values())
        assert later["prevented_cents"] == later["incumbent"]["net_cents"] - world_view["net_cents"]
        # the order's amount and the exposure the evidence states, by hand
        amount = int(tables["order_attempts"].set_index("order_id").loc[order, "amount_cents"])
        down = amount * terms["down_payment_bps"] // 10_000
        settled = amount - (amount * terms["merchant_discount_bps"] + 5_000) // 10_000
        row = facts["evidence"]["row"]
        assert row["amount_cents"] == amount
        assert row["order_exposure_cents"] == settled - down
        assert world_view["cash_by_kind"]["merchant_settlement"] == -settled
        assert [c["amount_cents"] for c in world_view["cash"]
                if c["kind"] == "customer_payment"][0] == down


def test_what_the_incumbent_did_shows_in_its_cash(declined_facts, reviewed_facts):
    # declined at checkout: no order, no cash; the world's loss is what was prevented
    takeover = alert(declined_facts, "account_takeover")["later"]
    assert takeover["fate"]["route"] == "auto_decline"
    assert takeover["incumbent"]["cash"] == [] and takeover["incumbent"]["net_cents"] == 0
    assert takeover["prevented_cents"] == -README_NET[12]
    assert [e["event"] for e in takeover["incumbent"]["events"]] == ["checkout"]
    # held, then voided before shipment: the down payment is refunded, nothing else moves
    stolen = alert(reviewed_facts, "card_testing")["later"]
    assert stolen["fate"]["void_cause"] in ("decline", "hold_cancelled")
    down = 65_000 * _terms()["down_payment_bps"] // 10_000
    assert [(c["kind"], c["amount_cents"]) for c in stolen["incumbent"]["cash"]] == [
        ("customer_payment", down), ("refund", -down)]
    assert stolen["incumbent"]["net_cents"] == 0 and stolen["approve_all"]["differs"] is True
    # held and cleared: the same cash as approve-all, the later events moved by the hold
    cleared = alert(reviewed_facts, "account_takeover")["later"]
    assert cleared["fate"]["hold_outcome"] == "cleared"
    assert cleared["incumbent"]["net_cents"] == README_NET[12] and cleared["prevented_cents"] == 0


# ------------------------------------------------------------------ latent truth, apart


def test_latent_truth_is_a_separate_block(declined_facts):
    facts = alert(declined_facts, "account_takeover")
    assert facts["latent"]["order"]["pattern_id"] == "P-ATO"
    outside = {key: value for key, value in facts.items() if key != "latent"}
    text = json.dumps(outside)
    for word in ("P-ATO", "pattern_id", "intent", "episode", "actor", "mimic"):
        assert word not in text


# ------------------------------------------------------------------ stable files


def test_a_rebuild_writes_the_same_bytes(declines, declined_facts, tmp_path):
    first = facts_module.write(declined_facts, tmp_path / "a")
    assert select_module.main(["--run", str(declines), "--out", str(tmp_path / "b"),
                               "--no-changes"]) == 0
    for path in first:
        again = tmp_path / "b" / path.name
        assert path.read_bytes() == again.read_bytes()
        text = path.read_text()
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True,
                                  ensure_ascii=False) + "\n"
        assert "NaN" not in text and "/Users/" not in text


def test_plain_values_are_rounded_and_timed_as_text():
    assert facts_module.plain({"b": np.float64(1 / 3), "a": np.int64(7), "c": np.nan,
                               "d": pd.Timestamp("2025-06-01 12:00:01"), "e": pd.NaT,
                               "f": [np.bool_(True), None]}) == {
        "a": 7, "b": 0.3333, "c": None, "d": "2025-06-01 12:00:01", "e": None,
        "f": [True, None]}


# ------------------------------------------------------------------ refusals


def test_evidence_assembled_after_the_action_is_refused():
    decision = {"kind": "review", "checkout_at": "2025-06-01 10:00:00",
                "evidence_at": "2025-06-02 00:00:00", "analyst_started_at": "2025-06-01 11:00:00",
                "action_at": "2025-06-01 11:05:00"}
    with pytest.raises(RuleError, match="between the checkout and the action"):
        facts_module._check_times(1, decision, {"decision_at": decision["evidence_at"]})
    with pytest.raises(RuleError, match="not assembled at evidence_at"):
        facts_module._check_times(1, {**decision, "action_at": "2025-06-02 08:00:00"},
                                  {"decision_at": "2025-06-02 01:00:00"})


def test_a_band_the_thresholds_do_not_give_is_refused(declines):
    run = facts_module.load_run(declines)
    run.decline_threshold = 60.0  # the takeover's 45 would then go to review
    with pytest.raises(RuleError, match="routes to review"):
        facts_module.build(run)


def test_a_disposition_the_saved_row_does_not_give_is_refused(reviews):
    run = facts_module.load_run(reviews)
    run.reviews = run.reviews.assign(disposition="clear")
    with pytest.raises(RuleError, match="the saved row gives hold"):
        facts_module.build(run)


def test_a_run_without_the_tuned_incumbent_or_its_frames_is_refused(declines, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(declines, copy)
    tune = json.loads((copy / "results" / "tune.json").read_text())
    tune["tables"]["tune.chosen"][0]["chosen_version"] = "elsewhere"
    (copy / "results" / "tune.json").write_text(json.dumps(tune))
    with pytest.raises(RuleError, match="policy versions"):
        facts_module.load_run(copy)
    tune["tables"]["tune.chosen"] = tune["tables"]["tune.chosen"][1:]
    (copy / "results" / "tune.json").write_text(json.dumps(tune))
    with pytest.raises(RuleError, match="no tuned incumbent_rules"):
        facts_module.load_run(copy)
    (copy / "worlds" / "416-baseline" / "checkout_rows.pkl").unlink()
    with pytest.raises(RuleError, match="checkout_rows.pkl"):
        facts_module.load_run(copy)


# ------------------------------------------------------------------ memo packets


def test_memo_inputs_are_the_saved_rows_of_the_selected_alerts(reviews, reviewed_facts):
    frame = facts_module.memo_inputs(reviews)
    assert frame["slot"].tolist() == ["account_takeover", "never_pay", "card_testing"]
    for record in frame.to_dict("records"):
        facts = alert(reviewed_facts, record["file"], record["slot"])
        assert record["alert_id"] == facts["publication"]["alert_id"]
        assert str(record["evidence_at"]) == facts["decision"]["evidence_at"]
        assert record["decision_at"] == record["evidence_at"]
        assert facts_module.plain({c: record[c] for c in facts_module.ROW_COLUMNS}) == \
            facts["evidence"]["row"]


# ------------------------------------------------------------------ a configured run


RUN = os.environ.get("BNPL_CASES_RUN")


@pytest.mark.skipif(not RUN, reason="set BNPL_CASES_RUN to a pipeline run directory")
def test_a_configured_run_reconciles_with_its_world():
    run = facts_module.load_run(Path(RUN))
    built = facts_module.build(run)
    tables = run.tables
    cash_events = tables["cash_events"]
    for entry in built.values():
        for facts in entry["alerts"].values():
            if not facts["selected"]:
                continue
            order = facts["publication"]["order_id"]
            row, decision = facts["evidence"]["row"], facts["decision"]
            at = pd.Timestamp(decision["evidence_at"])
            checkout, action = (pd.Timestamp(decision[k]) for k in ("checkout_at", "action_at"))
            assert checkout <= at <= action
            # outcome counts never exceed what the world had shown by then
            hand = _hand_history(tables, row["user_id"], at)
            for key, value in hand.items():
                assert row[key] <= value, (order, key)
            rows = cash_events.loc[cash_events["order_id"] == order]
            assert facts["later"]["approve_all"]["net_cents"] == int(rows["amount_cents"].sum())
            amount = int(tables["order_attempts"].set_index("order_id").loc[order, "amount_cents"])
            assert row["amount_cents"] == amount
