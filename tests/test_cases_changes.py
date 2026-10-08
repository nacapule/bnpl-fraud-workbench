"""Tested changes: each declared condition fires exactly where its words say, the variant
routes and reviews by it and nothing else, the replay is the case cell's, and the
published effect is the difference of the results' own measures."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from test_cases_facts import make_run

from cases import changes as changes_module
from cases import facts as facts_module
from cases.changes import CHANGES, VariantReviewer
from core import asof, config, world
from core import protocol as protocol_module
from core.actions import Check, Disposition
from queue_sim import stage
from queue_sim.reviewer import Reviewer

MINI = "tests/fixtures/mini_world"
WINDOW = (pd.Timestamp("2024-12-01"), pd.Timestamp("2025-07-01"))


@pytest.fixture(scope="module", autouse=True)
def mini_window():
    """The mini world's orders fall before the protocol's test window: the case cell's
    window is the mini world's whole order span here."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(facts_module.Run, "window", property(lambda self: WINDOW))
        yield


@pytest.fixture(scope="module")
def tables():
    return world.read_world(MINI)


@pytest.fixture(scope="module")
def quiet(tables) -> dict:
    """A context row on which no FP-2 condition holds (the mini world's first order)."""
    context = asof.build_context(tables)
    row = context.loc[context["order_id"] == 1].iloc[0].to_dict()
    assert changes_module.engine.score(pd.DataFrame([row]))[0] == 0
    return row


def rows(quiet, *overrides) -> pd.DataFrame:
    return pd.DataFrame([{**quiet, **o} for o in overrides])


def change(file):
    return next(c for c in CHANGES if c.file == file)


R03 = {"bin_ip_country_mismatch": 1, "avs_mismatch": 1}
TAKEOVER = {"account_age_days": 512.0, "device_link_age_hours": 0.24,
            "ship_address_first_use_age_hours": 0.0, "ship_to_home": 0}
HOME = {"device_link_age_hours": 5_370.0, "ship_to_home": 1, "home_address_age_days": 224.0}


# ------------------------------------------------------------------ the declared conditions


def test_each_file_has_one_declared_change():
    assert [c.file for c in CHANGES] == ["account_takeover", "never_pay_vs_hardship",
                                         "traveller", "card_testing", "ring"]
    for c in CHANGES:
        assert c.change and c.motivation and c.mechanism and c.parameters


def test_new_device_and_drop_address_on_an_established_account(quiet):
    frame = rows(quiet, TAKEOVER, {**TAKEOVER, "account_age_days": 90.0},
                 {**TAKEOVER, "account_age_days": 89.9},
                 {**TAKEOVER, "device_link_age_hours": 72.0},
                 {**TAKEOVER, "device_link_age_hours": 72.1},
                 {**TAKEOVER, "ship_address_first_use_age_hours": 0.5},
                 {**TAKEOVER, "ship_to_home": 1}, {**TAKEOVER, "device_link_age_hours": np.nan})
    assert changes_module.new_device_drop_address(frame).tolist() == [
        True, True, False, True, False, False, False, False]


def test_a_second_plan_before_any_installment_is_due(quiet):
    frame = rows(quiet, {"installments_due_user": 0, "open_balance_user_cents": 11_808},
                 {"installments_due_user": 1, "open_balance_user_cents": 11_808},
                 {"installments_due_user": 0, "open_balance_user_cents": 0})
    assert changes_module.stacked_plan(frame).tolist() == [True, False, False]


def test_r03_on_a_known_device_shipping_home(quiet):
    frame = rows(quiet, {**R03, **HOME}, {**HOME},  # no R03: nothing to except
                 {**R03, **HOME, "device_link_age_hours": 2_159.0},
                 {**R03, **HOME, "device_link_age_hours": 2_160.0},
                 {**R03, **HOME, "ship_to_home": 0}, {**R03, **HOME, "home_address_age_days": 89.0})
    assert changes_module.known_device_home(frame).tolist() == [
        True, False, False, True, False, False]


def test_one_device_decline_on_a_new_account(quiet):
    frame = rows(quiet, {"processor_declines_device_24h": 1, "account_age_days": 0.02},
                 {"processor_declines_device_24h": 1, "account_age_days": 7.0},
                 {"processor_declines_device_24h": 0, "account_age_days": 0.02},
                 {"processor_declines_device_24h": 3, "account_age_days": 0.02})  # R07 holds
    assert changes_module.new_account_device_decline(frame).tolist() == [
        True, False, False, False]


def test_linkage_only(quiet):
    frame = rows(quiet, {"accounts_on_device_30d": 3}, {"accounts_on_device_30d": 3, **R03},
                 {}, {"accounts_on_address_30d": 3, "email_root_other_accounts": 1},
                 {"accounts_on_device_30d": 3, "email_domain_class": 2})  # R06(a) is Context
    assert changes_module.linkage_only(frame).tolist() == [True, False, False, True, False]


# ------------------------------------------------------------------ routing and review


def _policy(file, incumbent):
    policy = change(file).policy(incumbent)
    assert (policy.review_threshold, policy.decline_threshold) == (35.0, 40.0)
    return policy


@pytest.fixture(scope="module")
def incumbent():
    from test_cases_facts import _incumbent

    return _incumbent(35.0, 40.0)


@pytest.mark.parametrize("file, override, review, decline, route", [
    ("account_takeover", TAKEOVER, 35, 35, "review"),
    ("account_takeover", {**TAKEOVER, **R03}, 70, 70, "auto_decline"),
    ("never_pay_vs_hardship", {"installments_due_user": 0, "open_balance_user_cents": 500},
     1000, 1000, "auto_decline"),
    ("traveller", {**R03, **HOME}, 0, 0, "approve"),
    ("traveller", {**R03, **HOME, "device_link_age_hours": 10.0}, 35, 35, "review"),
    ("card_testing", {"processor_declines_device_24h": 1, "account_age_days": 0.02}, 40, 40,
     "auto_decline"),
    ("ring", {"accounts_on_device_30d": 3}, 40, 0, "review"),
    ("ring", {"accounts_on_device_30d": 3, **R03}, 75, 75, "auto_decline"),
])
def test_variants_route_by_the_declared_condition(quiet, incumbent, file, override, review,
                                                  decline, route):
    policy = _policy(file, incumbent)
    routed = policy.route(rows(quiet, override))
    assert (routed["review_score"].iloc[0], routed["decline_score"].iloc[0]) == (review, decline)
    assert routed["route"].iloc[0] == route
    assert incumbent.route(rows(quiet, {}))["route"].iloc[0] == "approve"


def test_the_reviewer_reads_the_new_conditions(quiet):
    takeover = {**quiet, **TAKEOVER}
    assert Reviewer().decide(1, takeover, ()).disposition is Disposition.CLEAR
    decision = change("account_takeover").reviewer().decide(1, takeover, ())
    assert decision.disposition is Disposition.HOLD
    assert decision.checks_to_run == (Check.CONTACT,)
    traveller = {**quiet, **R03, **HOME}
    held = Reviewer().decide(1, traveller, ())
    assert held.disposition is Disposition.HOLD and held.checks_to_run == (Check.ID_CHECK,)
    assert change("traveller").reviewer().decide(1, traveller, ()).disposition \
        is Disposition.CLEAR
    for file in ("never_pay_vs_hardship", "card_testing", "ring"):
        assert not isinstance(change(file).reviewer(), VariantReviewer)


def test_the_replay_is_the_stage_benchs(tables, incumbent):
    bench = stage.Bench.of(tables, asof.build_context(tables), seed=416,
                           observed_until=pd.Timestamp("2025-06-30 23:59:59"))
    expected = bench.run(incumbent, WINDOW, stage.base_staffing(bench.policy_cfg))
    got = changes_module.replay(bench, incumbent, Reviewer(), WINDOW)
    pd.testing.assert_frame_equal(got.fates, expected.fates)
    pd.testing.assert_frame_equal(got.reviews, expected.reviews)


# ------------------------------------------------------------------ the published effect


@pytest.fixture(scope="module")
def tested(tables, tmp_path_factory):
    run = facts_module.load_run(make_run(tmp_path_factory.mktemp("changes"), tables, 35.0, 40.0))
    built = facts_module.build(run)
    return run, built, changes_module.run_changes(run, built, check_declared=False)


def _rule_net(world_block, raw) -> Fraction:
    """The recommendation rule's net contribution by hand: ledger net, less the LTV proxy
    per legitimate customer lost, less the analyst allotment at the loaded hourly cost."""
    ltv = round(float(config.load("policy")["costs"]["false_decline_ltv_usd"]) * 100)
    hourly = Fraction(str(raw["capacity"]["analyst_loaded_hourly_usd"])) * 100
    adjudicated = world_block["adjudicated"]
    lost = adjudicated["legitimate_declined"] + adjudicated["legitimate_cancelled"]
    return (world_block["cash"]["net_cents"] - ltv * lost
            - Fraction(world_block["review"]["available_minutes"]) * hourly / 60)


def test_every_file_records_its_declared_change_and_one_run(tested):
    _, _, blocks = tested
    assert set(blocks) == {c.file for c in CHANGES}
    for c in CHANGES:
        block = blocks[c.file]
        assert {k: block[k] for k in ("name", "change", "motivation", "mechanism")} == {
            "name": c.name, "change": c.change, "motivation": c.motivation,
            "mechanism": c.mechanism}
        assert block["superseded"] == [] and "development world" in block["illustration"]
        assert block["declared_for_order"] == c.declared_for
        assert block["noted_after_run"] == c.noted_after_run


def test_the_effect_is_the_difference_of_the_results_measures(tested):
    run, _, blocks = tested
    raw = protocol_module.load_protocol().raw
    for block in blocks.values():
        result = block["result"]
        before, after, diff = (result["world"][k] for k in ("incumbent", "variant", "difference"))
        assert before["orders"] == after["orders"] == len(run.fates)
        for group in ("cash", "adjudicated", "latent", "review"):
            for column, value in diff[group].items():
                assert value == pytest.approx(after[group][column] - before[group][column])
        for measured in (before, after):
            assert measured["cash"]["rule_net_cents"] == pytest.approx(
                float(_rule_net(measured, raw)), abs=1e-3)
        per_1000 = (after["cash"]["rule_net_cents"] - before["cash"]["rule_net_cents"]) \
            * 1000 / after["orders"]
        assert diff["rule_net_vs_incumbent_per_1000_orders_cents"] == pytest.approx(per_1000)
        assert before["policy_version"] == run.policy_version != after["policy_version"]


def test_the_case_orders_are_shown_under_both_policies(tested):
    _, built, blocks = tested
    # the takeover (score 45) is declined either way; its record says it did not change
    takeover = blocks["account_takeover"]["result"]["case_order"]
    assert takeover["order_id"] == 12 and takeover["changed"] is False
    assert takeover["incumbent"]["route"] == takeover["variant"]["route"] == "auto_decline"
    # eve's stolen card: 2 device declines on a new account now decline at checkout
    stolen = blocks["card_testing"]["result"]["case_order"]
    assert stolen["incumbent"]["route"] == "review" and stolen["variant"]["route"] == \
        "auto_decline" and stolen["changed"] is True
    assert stolen["variant"]["net_cents"] == 0
    assert stolen["incumbent"]["net_cents"] == built["card_testing"]["alerts"]["card_testing"][
        "later"]["incumbent"]["net_cents"]
    # files whose slot is missing in this world still get the world's effect
    assert blocks["traveller"]["result"]["case_order"] is None
    assert blocks["traveller"]["result"]["account_orders"] == []


def test_the_incumbent_must_be_the_runs(tables, tmp_path):
    run = facts_module.load_run(make_run(tmp_path, tables, 35.0, 40.0))
    run.fates = run.fates.assign(route=run.fates["route"].replace({"review": "approve"}))
    with pytest.raises(changes_module.RuleError, match="differs from the run's"):
        changes_module.run_changes(run, {}, check_declared=False)


def test_a_change_runs_only_for_the_order_it_was_declared_for(tables, tmp_path):
    run = facts_module.load_run(make_run(tmp_path, tables, 35.0, 40.0))
    built = facts_module.build(run)  # the mini world selects other orders
    with pytest.raises(changes_module.RuleError, match="declared for order 136685; the "
                                                      "selection gives 12"):
        changes_module.run_changes(run, built)
    # a missing slot is not another order: the file keeps the change's effect on the world
    blocks = changes_module.run_changes(run, built, changes=(change("traveller"),))
    result = blocks["traveller"]["result"]
    assert result["selected_order"] is None and result["case_order"] is None
    assert result["world"]["difference"]["review"]["reviews"] <= 0


def test_the_rules_model_must_be_the_runs(tables, tmp_path):
    directory = make_run(tmp_path, tables, 35.0, 40.0)
    meta = directory / "fit" / "416" / "model_rules.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "version": "another"}))
    with pytest.raises(changes_module.RuleError, match="rebuilt incumbent"):
        changes_module.incumbent_policy(facts_module.load_run(directory))
