"""The as-of context against a row-by-row reading of its specification, and its history.

On the mini world and on seeded random worlds (``asof_worlds``), ``build_context``
equals the oracle (``asof_oracle``) at checkout and at later decisions, and
``outcome_columns`` equals it under random policies. Rows do not change when the
world is cut to what was known by an earlier time (also inside a second, and when
knowledge arrives after the event); labels and latent truth change nothing; and a
policy's declines, cancelled holds and released holds shape what later orders see.
"""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from asof_oracle import Oracle
from asof_worlds import (
    HOUR,
    MINUTE,
    Act,
    known_by,
    later_decisions,
    random_acts,
    random_blocks,
    random_world,
    realize,
    seconds,
)

from core import asof, world

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_world"
WORLDS = ("mini", 1, 2, 3, 4)
POLICY_SEEDS = (4, 5)
CUT_SEEDS = (7, 8)


@cache
def _world(name) -> dict[str, pd.DataFrame]:
    return world.read_world(FIXTURE) if name == "mini" else random_world(name)


def _decisions(pairs) -> pd.DataFrame:
    pairs = list(pairs)
    return pd.DataFrame({"order_id": [int(o) for o, _ in pairs],
                         "decision_at": pd.to_datetime([at for _, at in pairs]).astype(
                             "datetime64[s]")})


def _checkouts(tables) -> pd.DataFrame:
    orders = tables["order_attempts"]
    return _decisions(zip(orders["order_id"], orders["known_at"], strict=True))


@cache
def _world_rows(name) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """(label, build_context, oracle) at checkout and at later decisions."""
    tables = _world(name)
    oracle = Oracle(tables)
    at_checkout = asof.build_context(tables)
    later = later_decisions(tables, seed=17, count=150)
    return [(f"{name} at checkout", at_checkout,
             oracle.frame(at_checkout[["order_id", "decision_at"]])),
            (f"{name} later", asof.build_context(tables, later), oracle.frame(later))]


@cache
def _policy_rows(seed) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """(label, outcome_columns, oracle) under a random policy, at checkout, at later
    decisions and for a few accounts' decisions alone."""
    tables = random_world(seed)
    acts = random_acts(tables, seed)
    realized, state = realize(tables, acts, random_blocks(tables, seed))
    oracle = Oracle(realized, state)
    later = later_decisions(realized, seed=seed, count=200, acts=acts)
    few = later[later["order_id"].isin(later["order_id"].drop_duplicates().head(6))]
    return [(f"policy {seed} {label}", asof.outcome_columns(realized, state, decisions),
             oracle.frame(decisions))
            for label, decisions in (("at checkout", _checkouts(realized)), ("later", later),
                                     ("few", few))]


def _assert_column(column: str, compared) -> None:
    problems = []
    for label, actual, expected in compared:
        assert actual["order_id"].tolist() == expected["order_id"].tolist(), label
        got, want = actual[column].to_numpy(), expected[column].to_numpy()
        if want.dtype.kind == "f":
            same = np.isclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            same = got == want
        problems += [f"{label}: order {o} at {at}: context {g}, oracle {w}" for o, at, g, w in zip(
            actual["order_id"][~same], actual["decision_at"][~same], got[~same], want[~same],
            strict=True)]
    assert not problems, f"{column}: {len(problems)} rows differ, e.g. " + "; ".join(problems[:4])


# ------------------------------------------------------------------ the oracle
@pytest.mark.parametrize("column", asof.COLUMN_NAMES)
def test_world_context_matches_the_oracle(column: str) -> None:
    _assert_column(column, [rows for name in WORLDS for rows in _world_rows(name)])


@pytest.mark.parametrize("column", asof.OUTCOME_COLUMNS)
def test_outcome_columns_under_a_policy_match_the_oracle(column: str) -> None:
    _assert_column(column, [rows for seed in POLICY_SEEDS for rows in _policy_rows(seed)])


def test_rows_follow_the_decisions() -> None:
    """By default every attempt at its checkout, in the event order; otherwise as asked."""
    tables = _world(1)
    (_, at_checkout, _), (_, later, _) = _world_rows(1)
    orders = tables["order_attempts"].sort_values(["known_at", "event_id"])
    assert at_checkout["order_id"].tolist() == orders["order_id"].tolist()
    assert (at_checkout["decision_at"].to_numpy() == orders["known_at"].to_numpy()).all()
    asked = later_decisions(tables, seed=17, count=150)
    assert later[["order_id", "decision_at"]].equals(asked)


def test_random_worlds_hold_the_cases_the_context_must_get_right() -> None:
    found: dict[str, int] = {}

    def count(name: str, value) -> None:
        found[name] = found.get(name, 0) + int(value)

    for seed in WORLDS[1:]:
        t = _world(seed)
        orders = t["order_attempts"]
        second = pd.DataFrame({"s": seconds(orders["known_at"]), "user": orders["user_id"],
                               "device": orders["device_id"]})
        count("attempts of one account in one second", second.duplicated(["s", "user"]).sum())
        count("attempts of two accounts on one device in one second",
              (second.groupby(["s", "device"])["user"].nunique() > 1).sum())
        events = np.concatenate([seconds(t[name]["known_at"]) for name in (
            "account_events", "fulfilments", "deliveries", "dispute_openings")])
        count("attempts in the second of another kind of event",
              np.isin(second["s"], events).sum())
        for name in ("dispute_openings", "dispute_resolutions", "deliveries",
                     "payment_reversals"):
            count(f"{name} known after they happened",
                  (t[name]["known_at"] > t[name]["occurred_at"]).sum())
        links = t["device_links"]
        count("devices linked at the second another link ends",
              links.merge(links, on="device_id").pipe(
                  lambda m: (m["created_at_x"] == m["removed_at_y"]).sum()))
        count("removed address links", t["address_links"]["removed_at"].notna().sum())
        homes = t["address_links"][t["address_links"]["role"] == "home"]
        count("homes of two accounts", (homes.groupby("address_id")["user_id"].nunique() > 1).sum())
        changes = t["account_events"][t["account_events"]["kind"] == "email_change"]
        count("two email changes in one second", changes.duplicated(["user_id", "known_at"]).sum())
        history = asof.email_history(t["accounts"], t["account_events"])
        count("email changed back", history.duplicated(["user_id", "email_root"]).sum())
        raw = pd.concat([t["accounts"][["user_id", "email"]], changes[["user_id", "email"]]])
        raw = raw.assign(root=raw["email"].map(asof.normalize_email))
        count("one normalized email written differently by two accounts",
              (raw.groupby("root").agg(users=("user_id", "nunique"), spellings=("email", "nunique"))
               .pipe(lambda g: ((g["users"] > 1) & (g["spellings"] > 1)).sum())))
        count("processor declines", (orders["processor_result"] == "declined").sum())
        first_only = t["promotions"].set_index("promo_id")["first_purchase_only"]
        uses = orders["promo_id"].dropna().map(first_only)
        count("first-purchase promotion uses", uses.sum())
        count("other promotion uses", (~uses.astype(bool)).sum())
        payments = t["payment_attempts"]
        count("failed payments", (payments["result"] == "failed").sum())
        count("payment retries", (payments["attempt_no"] > 1).sum())
        reversed_ = t["payment_reversals"].merge(payments, left_on="payment_event_id",
                                                 right_on="event_id")
        count("partial reversals",
              (reversed_["amount_cents_x"] < reversed_["amount_cents_y"]).sum())
        for name in ("plan_writeoffs", "victim_reports"):
            count(name, len(t[name]))
        approved = orders[orders["processor_result"] == "approved"].merge(
            t["merchants"], on="merchant_id")["category"].value_counts()
        count("categories with a reference", (approved >= asof.CATEGORY_MIN_SAMPLE).sum())
        count("categories without", (approved < asof.CATEGORY_MIN_SAMPLE).sum())
    assert all(found.values()), {name: n for name, n in found.items() if not n}


# ------------------------------------------------------------ prefix invariance
def _known_rows(tables, decisions, cut_second: int, place=None) -> None:
    """Rows of decisions that see nothing after the cut equal the cut world's rows."""
    full = asof.build_context(tables, decisions)
    order_place = {o: (s, world.EVENT_RANK["order_attempts"], e) for o, s, e in zip(
        tables["order_attempts"]["order_id"], seconds(tables["order_attempts"]["known_at"]),
        tables["order_attempts"]["event_id"], strict=True)}
    decided = seconds(decisions["decision_at"])
    checkout = np.array([decided[i] == order_place[o][0]
                         for i, o in enumerate(decisions["order_id"])])
    if place is None:
        seen = decided <= cut_second
    else:  # a checkout up to the place, a later decision before the place's second
        seen = np.where(checkout, [order_place[o] <= place for o in decisions["order_id"]],
                        decided < cut_second)
    assert seen.sum() > 20
    cut = known_by(tables, place if place is not None else
                   pd.Timestamp(cut_second, unit="s"))
    kept = decisions[seen].reset_index(drop=True)
    pd.testing.assert_frame_equal(asof.build_context(cut, kept),
                                  full[seen].reset_index(drop=True), check_exact=True)


@pytest.mark.parametrize("seed", CUT_SEEDS)
def test_rows_do_not_change_when_the_world_is_cut_at_a_time(seed: int) -> None:
    tables = random_world(seed)
    decisions = pd.concat([_checkouts(tables), later_decisions(tables, seed, 120)],
                          ignore_index=True)
    rng = random.Random(seed)
    decided = sorted(set(seconds(decisions["decision_at"])))  # decisions at the cut itself
    for cut in (*rng.sample(decided[20:], 5), rng.randint(int(decided[20]), int(decided[-1]))):
        _known_rows(tables, decisions, int(cut))


@pytest.mark.parametrize("seed", CUT_SEEDS)
def test_rows_do_not_change_when_the_world_is_cut_inside_a_second(seed: int) -> None:
    tables = random_world(seed)
    decisions = pd.concat([_checkouts(tables), later_decisions(tables, seed, 120)],
                          ignore_index=True)
    events = pd.concat([pd.DataFrame({"s": seconds(tables[name]["known_at"]),
                                      "rank": world.EVENT_RANK[name],
                                      "event_id": tables[name]["event_id"]})
                        for name in world.EVENT_RANK], ignore_index=True)
    events = events.sort_values(["s", "rank", "event_id"])
    attempts_in = events[events["rank"] == world.EVENT_RANK["order_attempts"]].groupby("s").size()
    crowded = attempts_in[attempts_in > 1].index  # seconds with several attempts
    rng = random.Random(seed)
    for second in rng.sample(sorted(crowded), 3):
        inside = events[events["s"] == second]
        pivot = inside.iloc[rng.randrange(len(inside) - 1)]  # never the second's last event
        _known_rows(tables, decisions, int(second),
                    (int(second), int(pivot["rank"]), int(pivot["event_id"])))


def _state_by(state: asof.PolicyState, second: int) -> asof.PolicyState:
    """What the policy had done by the end of ``second``: later approvals, voids, holds and
    blocks dropped, and a hold released later still pending."""
    def by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        return frame[seconds(frame[column]) <= second].reset_index(drop=True)

    held = by(state.held, "held_at")
    pending = seconds(held["released_at"]) > second
    held.loc[pending, "released_at"] = pd.NaT
    held.loc[pending, "outcome"] = None
    return asof.PolicyState(approved=by(state.approved, "approved_at"),
                            voided=by(state.voided, "at"), blocked=by(state.blocked, "at"),
                            held=held)


@pytest.mark.parametrize("seed", POLICY_SEEDS)
def test_policy_rows_do_not_change_when_the_policy_is_cut_at_a_time(seed: int) -> None:
    """Under a random policy, outcome columns at decisions up to a time equal those from
    the realized tables and the policy's state as they stood then, cut just before and
    at holds' releases and voids and at random decision times."""
    tables = random_world(seed)
    acts = random_acts(tables, seed)
    realized, state = realize(tables, acts, random_blocks(tables, seed))
    decisions = pd.concat([_checkouts(realized), later_decisions(realized, seed, 200, acts)],
                          ignore_index=True)
    full = asof.outcome_columns(realized, state, decisions)
    rng = random.Random(seed)
    released = sorted(set(seconds(state.held["released_at"].dropna())))
    voids = sorted(set(seconds(state.voided["at"])))
    decided = seconds(decisions["decision_at"])
    cuts = {int(s) + step for s in (*rng.sample(released, 4), *rng.sample(voids, 3))
            for step in (-1, 0)}
    cuts |= set(rng.sample(sorted(set(decided[decided > decided.min()].tolist())), 4))
    for cut in sorted(cuts):
        seen = decided <= cut
        assert seen.sum() > 20
        rows = asof.outcome_columns(known_by(realized, pd.Timestamp(cut, unit="s")),
                                    _state_by(state, cut), decisions[seen].reset_index(drop=True))
        pd.testing.assert_frame_equal(rows, full[seen].reset_index(drop=True),
                                      check_exact=True, obj=f"policy {seed} cut at {cut}")


def _late_dispute():
    """A world, an item-not-received dispute notified at least two days after it was
    filed, a time between the two, and another order of the account placed by then."""
    for seed in range(10):
        tables = random_world(seed)
        orders = tables["order_attempts"]
        disputes = tables["dispute_openings"].merge(orders[["order_id", "user_id"]],
                                                    on="order_id")
        late = disputes["known_at"] - disputes["occurred_at"] >= pd.Timedelta(days=2)
        for dispute in disputes[late & (disputes["reason"] == "item_not_received")].itertuples():
            cut = (dispute.occurred_at + (dispute.known_at - dispute.occurred_at) / 2).floor("s")
            other = orders[(orders["user_id"] == dispute.user_id)
                           & (orders["order_id"] != dispute.order_id)
                           & (orders["known_at"] <= cut)]
            if len(other):
                return tables, dispute, cut, int(other["order_id"].iloc[-1])
    raise AssertionError("no world with a late-notified item-not-received dispute")


def test_knowledge_that_arrives_after_a_cut_was_not_visible_before_it() -> None:
    """An item-not-received dispute filed before the cut but notified after it is not in
    the cut world, and the account's other order did not see it at the cut; once known,
    it counts."""
    tables, dispute, cut, order = _late_dispute()
    cut_world = known_by(tables, cut)
    assert dispute.dispute_id not in set(cut_world["dispute_openings"]["dispute_id"])
    rows = asof.build_context(tables, _decisions([(order, cut), (order, dispute.known_at)]))
    assert rows["inr_disputes_opened_user"].iloc[1] > rows["inr_disputes_opened_user"].iloc[0]
    pd.testing.assert_frame_equal(asof.build_context(cut_world, _decisions([(order, cut)])),
                                  rows.iloc[:1], check_exact=True)


# ------------------------------------------------------- labels and latent truth
@pytest.mark.parametrize("name", ("mini", 2))
def test_labels_and_latent_truth_change_no_context_value(name) -> None:
    tables = _world(name)
    rng = np.random.default_rng(0)
    decisions = pd.concat([_checkouts(tables), later_decisions(tables, 5, 60)], ignore_index=True)
    base = asof.build_context(tables, decisions)
    state = asof.PolicyState.approve_all(tables)
    labels = tables["labels"]
    observable = {n: f for n, f in tables.items()
                  if n != "labels" and n not in world.LATENT_TABLES}
    flipped_truth = {
        "latent_accounts": tables["latent_accounts"].assign(actor="fraudster"),
        "latent_orders": tables["latent_orders"].assign(intent="fraud", pattern_id="P-ATO"),
        "latent_episodes": tables["latent_episodes"].iloc[:0]}
    variants = {
        "labels flipped": {**tables, "labels": labels.assign(
            label=1 - labels["label"], basis=rng.permutation(labels["basis"].to_numpy()))},
        "labels deleted": {**tables, "labels": labels.iloc[:0]},
        "labels rewritten": {**tables, "labels": labels.assign(
            order_id=rng.permutation(labels["order_id"].to_numpy()),
            label_known_at=labels["label_known_at"] - pd.Timedelta(days=45))},
        "latent truth rewritten": {**tables, **flipped_truth},
        "labels and latent truth absent": observable,
    }
    for label, mutated in variants.items():
        pd.testing.assert_frame_equal(asof.build_context(mutated, decisions), base,
                                      check_exact=True, obj=label)
        pd.testing.assert_frame_equal(asof.attempt_columns(mutated, decisions),
                                      base[[*asof.KEY_COLUMNS, *asof.ATTEMPT_COLUMNS]],
                                      check_exact=True, obj=label)
        pd.testing.assert_frame_equal(asof.outcome_columns(mutated, state, decisions),
                                      base[[*asof.KEY_COLUMNS, *asof.OUTCOME_COLUMNS]],
                                      check_exact=True, obj=label)


# ------------------------------------------------- the policy's own history
# In the mini world ana (account 3) orders twice: order 4 on 2025-01-05 19:20 ($84.50,
# repaid on each due date, shipped 12 hours after checkout) and order 14 on 2025-02-10.
FIRST, NEXT = 4, 14
HISTORY = ("approved_orders_user_ever", "approved_orders_user_24h", "installments_due_user",
           "installments_paid_user", "installments_failed_user", "installments_paid_share_user",
           "open_balance_user_cents", "promo_redemptions_user")


def _next_order(acts: dict[int, Act], at: pd.Timestamp | None = None) -> pd.Series:
    """Order 14's outcome columns under a policy that did ``acts`` (at checkout by default)."""
    mini = _world("mini")
    realized, state = realize(mini, acts)
    when = mini["order_attempts"].set_index("order_id").loc[NEXT, "known_at"] if at is None else at
    return asof.outcome_columns(realized, state, _decisions([(NEXT, when)])).iloc[0]


def test_the_next_order_sees_the_first_under_approve_all() -> None:
    row = _next_order({})
    assert (row["approved_orders_user_ever"], row["installments_due_user"],
            row["installments_paid_user"]) == (1, 2, 2)


def test_a_declined_first_order_leaves_the_next_without_history() -> None:
    row = _next_order({FIRST: Act("decline")})
    assert (row[list(HISTORY)] == 0).all(), row[list(HISTORY)].to_dict()


def test_a_held_then_cancelled_first_order_leaves_the_next_without_history() -> None:
    cancelled = {FIRST: Act("hold", 10 * MINUTE, 48 * HOUR, "cancelled")}
    row = _next_order(cancelled)
    assert (row[list(HISTORY)] == 0).all(), row[list(HISTORY)].to_dict()
    mini = _world("mini")
    realized, state = realize(mini, cancelled)
    checkout = mini["order_attempts"].set_index("order_id").loc[FIRST, "known_at"]
    during, after = checkout + pd.Timedelta(hours=1), checkout + pd.Timedelta(hours=49)
    own = asof.outcome_columns(realized, state, _decisions([(FIRST, during), (FIRST, after)]))
    assert own["cancelled_at_decision"].tolist() == [0, 1]
    assert own["shipped_at_decision"].tolist() == [0, 0]


def test_a_hold_released_later_moves_the_due_dates_the_next_order_sees() -> None:
    mini = _world("mini")
    released = {FIRST: Act("hold", 10 * MINUTE, 48 * HOUR, "cleared")}
    realized, _ = realize(mini, released)
    plan = mini["plans"].set_index("order_id").loc[FIRST, "plan_id"]

    def due(tables):
        schedule = tables["installment_schedule"]
        return schedule[schedule["plan_id"] == plan].set_index("seq")["due_at"]

    moved = due(realized) - due(mini)
    assert moved.tolist() == [pd.Timedelta(0)] + [pd.Timedelta(hours=48)] * 3
    last_due, moved_due = due(mini)[3], due(realized)[3]  # 2025-02-16 and -18, 19:20
    between = last_due + pd.Timedelta(days=1)
    due_and_paid = ["installments_due_user", "installments_paid_user"]
    assert _next_order({}, between)[due_and_paid].tolist() == [3, 3]
    assert _next_order(released, between)[due_and_paid].tolist() == [2, 2]
    just_before = _next_order(released, moved_due - pd.Timedelta(seconds=1))
    assert just_before["installments_due_user"] == 2
    assert _next_order(released, moved_due)["installments_due_user"] == 3
    assert _next_order(released)["approved_orders_user_ever"] == 1
