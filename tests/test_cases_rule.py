"""Case selection by the pre-registered rule: the protocol is read as written, candidates
follow the SHA-256 order, slots take reviewed alerts first, an order is used once, a
missing slot is reported with its counts, and nothing about what happened after the
decision (dispositions, labels, cash) changes a pick."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from cases import rule as rule_module
from cases.rule import RuleError, Term, load_rule, parse_predicate, population, select
from core import protocol as protocol_module

PREFIX = "bnpl-cases-2026-10:"


@pytest.fixture(scope="module")
def raw() -> dict:
    return protocol_module.load_protocol().raw


@pytest.fixture(scope="module")
def rule(raw):
    return load_rule(raw)


# ------------------------------------------------------------------ reading the rule


def test_the_frozen_rule_reads_as_registered(rule):
    assert (rule.seed, rule.family) == (416, "baseline")
    assert rule.prefix == PREFIX
    assert rule.slot_order == ("account_takeover", "never_pay", "hardship", "traveller",
                               "card_testing", "ring")
    assert {name: f.slots for name, f in rule.files.items()} == {
        "account_takeover": ("account_takeover",),
        "never_pay_vs_hardship": ("never_pay", "hardship"), "traveller": ("traveller",),
        "card_testing": ("card_testing",), "ring": ("ring",)}
    assert rule.files["never_pay_vs_hardship"].primary == "never_pay"
    assert rule.slots["card_testing"].tiers == ("checkout_alert",)
    assert rule.slots["traveller"].tiers == ("reviewed_alert", "checkout_alert")
    assert rule.slots["traveller"].terms == (
        Term("latent", "intent", "==", "legitimate"), Term("latent", "pattern_id", "is null", None),
        Term("mimic", "mimic", "has token", "travel"))
    assert rule.slots["card_testing"].terms == (
        Term("latent", "pattern_id", "==", "P-STOLEN"),
        Term("checkout", "processor_declines_device_24h", ">=", 1))


@pytest.mark.parametrize("path, text", [
    (("selection", "duplicate_rule"), "an order may fill two slots"),
    (("selection", "reviewed_rule"), "present in review_decisions with disposition clear"),
    (("population",), "every processor-approved checkout"),
    (("selection", "order"), "ascending order_id"),
])
def test_a_rule_worded_differently_is_refused(raw, path, text):
    changed = copy.deepcopy(raw)
    node = changed["cases"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = text
    with pytest.raises(RuleError):
        load_rule(changed)


def test_the_hash_prefix_is_read_from_the_protocol(raw):
    changed = copy.deepcopy(raw)
    changed["cases"]["selection"]["order"] = changed["cases"]["selection"]["order"].replace(
        PREFIX, "another-prefix:")
    assert load_rule(changed).prefix == "another-prefix:"


@pytest.mark.parametrize("text", ["latent_orders.pattern_id != P-ATO",
                                  "labels.label == 1", "pattern_id == P-ATO",
                                  "processor_declines_device_24h >= 1"])
def test_unknown_predicate_terms_are_refused(text):
    with pytest.raises(RuleError):
        parse_predicate(text, "latent_orders.intent == legitimate")


# ------------------------------------------------------------------ a constructed population

SPECS = [  # (pattern, intent, mimic, reviewed, device declines at checkout)
    ("P-ATO", "fraud", None, False, 0), ("P-ATO", "fraud", None, True, 0),
    ("P-ATO", "fraud", None, True, 0), ("P-NEVERPAY", "fraud", None, False, 0),
    (None, "legitimate", "hardship_default+travel", True, 0),
    (None, "legitimate", "travel", True, 0), (None, "legitimate", "travel", False, 0),
    (None, "legitimate", "traveller", True, 0),  # not the token "travel"
    ("P-PROMO", "abuse", "travel", True, 0),  # not legitimate
    ("P-STOLEN", "fraud", None, True, 2), ("P-STOLEN", "fraud", None, False, 0),
    ("P-STOLEN", "fraud", None, False, 1), ("P-SYNTH", "fraud", None, True, 0),
    (None, "legitimate", None, False, 0), (None, "legitimate", None, True, 0),
]


def world(specs=SPECS, *, first_order: int = 501):
    """Alerts, kept frames and latent rows for ``specs``, one alert per spec."""
    orders = np.arange(first_order, first_order + len(specs))
    checkout = pd.Timestamp("2025-06-02") + pd.to_timedelta(np.arange(len(specs)), "h")
    fates = pd.DataFrame({"order_id": orders, "user_id": orders + 1000, "checkout_at": checkout,
                          "route": ["review" if i % 3 else "auto_decline"
                                    for i in range(len(specs))]})
    checkout_rows = pd.DataFrame({"order_id": orders, "user_id": orders + 1000,
                                  "processor_declines_device_24h": [s[4] for s in specs]})
    latent = pd.DataFrame({"order_id": orders, "pattern_id": [s[0] for s in specs],
                           "episode_id": None, "intent": [s[1] for s in specs],
                           "mimic": [s[2] for s in specs]})
    reviewed = orders[[s[3] for s in specs]]
    reviews = pd.DataFrame({"order_id": reviewed, "taken_up_at": pd.Timestamp("2025-06-03"),
                            "decided_at": pd.Timestamp("2025-06-03 00:10"),
                            "disposition": "hold", "final": "clear"})
    return population(fates, checkout_rows, "v1"), reviews, checkout_rows, latent


def oracle(rule, alerts, reviews, checkout_rows, latent):
    """The rule applied by hand: hash order, explicit predicates, tiers, duplicates."""
    lat = latent.set_index("order_id")
    declines = checkout_rows.set_index("order_id")["processor_declines_device_24h"]
    done = set(reviews.loc[reviews["taken_up_at"].notna() & reviews["decided_at"].notna(),
                           "order_id"])

    def tokens(order):
        mimic = lat.loc[order, "mimic"]
        return (mimic if isinstance(mimic, str) else "").split("+")

    def legit(order):
        return lat.loc[order, "intent"] == "legitimate" and pd.isna(lat.loc[order, "pattern_id"])

    predicates = {
        "account_takeover": lambda o: lat.loc[o, "pattern_id"] == "P-ATO",
        "never_pay": lambda o: lat.loc[o, "pattern_id"] == "P-NEVERPAY",
        "hardship": lambda o: legit(o) and "hardship_default" in tokens(o),
        "traveller": lambda o: legit(o) and "travel" in tokens(o),
        "card_testing": lambda o: lat.loc[o, "pattern_id"] == "P-STOLEN" and declines[o] >= 1,
        "ring": lambda o: lat.loc[o, "pattern_id"] == "P-SYNTH",
    }
    ranked = sorted(int(o) for o in alerts["order_id"])
    ranked.sort(key=lambda o: (hashlib.sha256(f"{PREFIX}{o}".encode()).hexdigest(), o))
    taken, out = set(), {}
    for slot in rule.slot_order:
        pool = [o for o in ranked if predicates[slot](o) and o not in taken]
        choice = None
        for tier in rule.slots[slot].tiers:
            tiered = [o for o in pool if tier == "checkout_alert" or o in done]
            if tiered:
                choice = (tiered[0], tier)
                break
        if choice:
            taken.add(choice[0])
        out[slot] = choice
    return out


def picked(picks):
    return {slot: (p.order_id, p.tier) if p.selected else None for slot, p in picks.items()}


@pytest.mark.parametrize("first_order", [501, 7_001, 120_000, 3])
def test_picks_match_the_rule_applied_by_hand(rule, first_order):
    frames = world(first_order=first_order)
    assert picked(select(rule, *frames)) == oracle(rule, *frames)


def test_reviewed_alerts_come_first_and_card_testing_takes_any_alert(rule):
    alerts, reviews, rows, latent = world()
    picks = select(rule, alerts, reviews, rows, latent)
    reviewed = set(reviews["order_id"])
    for slot in ("account_takeover", "traveller"):
        assert picks[slot].tier == "reviewed_alert" and picks[slot].order_id in reviewed
    assert picks["never_pay"].tier == "checkout_alert"  # its only candidate was not reviewed
    assert picks["card_testing"].tier == "checkout_alert"
    assert picks["account_takeover"].tiers == {"reviewed_alert": 2, "checkout_alert": 3}
    assert (picks["account_takeover"].eligible, picks["account_takeover"].reviewed) == (3, 2)


def test_an_order_fills_one_slot_only(rule):
    # one legitimate order is both a hardship and a traveller candidate
    specs = [(None, "legitimate", "hardship_default+travel", True, 0),
             (None, "legitimate", "travel", True, 0)]
    alerts, reviews, rows, latent = world(specs)
    picks = select(rule, alerts, reviews, rows, latent)
    both, other = (int(o) for o in alerts["order_id"])
    assert picks["hardship"].order_id == both
    assert picks["traveller"].order_id == other
    assert (picks["traveller"].eligible, picks["traveller"].available) == (2, 1)


def test_a_missing_slot_is_reported_with_its_counts_and_never_filled(rule):
    specs = [(None, "legitimate", None, True, 0), ("P-STOLEN", "fraud", None, True, 0)]
    picks = select(rule, *world(specs))
    assert not any(p.selected for p in picks.values())
    card = picks["card_testing"]
    assert (card.order_id, card.tier, card.eligible, card.reviewed) == (None, None, 0, 0)


def test_what_happened_after_the_decision_never_changes_a_pick(rule):
    alerts, reviews, rows, latent = world()
    before = picked(select(rule, alerts, reviews, rows, latent))
    later = reviews.assign(disposition="clear", final="decline", checks_later=[[]] * len(reviews))
    assert picked(select(rule, alerts, later, rows, latent)) == before
    # an order taken up but not decided is not reviewed
    undecided = reviews.assign(decided_at=pd.NaT)
    picks = select(rule, alerts, undecided, rows, latent)
    assert all(p.tier in (None, "checkout_alert") for p in picks.values())


def test_the_population_is_the_alerted_orders_of_the_kept_frames():
    fates = pd.DataFrame({"order_id": [1, 2, 3, 4], "user_id": [9, 9, 8, 7],
                          "checkout_at": pd.to_datetime(["2025-06-01"] * 4),
                          "route": ["approve", "review", "blocked", "auto_decline"]})
    rows = pd.DataFrame({"order_id": [2, 4]})
    alerts = population(fates, rows, "abc")
    assert alerts["order_id"].tolist() == [2, 4]
    assert alerts["alert_id"].tolist() == ["2:abc", "4:abc"]
    assert alerts["band"].tolist() == ["review", "auto_decline"]
    with pytest.raises(RuleError):
        population(fates, pd.DataFrame({"order_id": [2, 3]}), "abc")


def test_the_selection_hash_is_sha256_of_the_prefix_and_decimal_id():
    assert rule_module.selection_hash(136685, PREFIX) == hashlib.sha256(
        b"bnpl-cases-2026-10:136685").hexdigest()
