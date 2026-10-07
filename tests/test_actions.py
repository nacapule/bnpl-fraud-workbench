"""What one policy's decisions do to the world's orders (core.actions), checked by hand.

The cases use the mini world (tests/fixtures/mini_world, README.md has each account's
story and net cash) and the product terms in config/world.yaml: 25% down rounded down,
three fortnightly installments, the merchant settled at shipment net of a 5% fee, a
$15 dispute fee, and a 20% recovery 30 days after the write-off. Expected values are
worked out from those terms and the fixture rows, never from the code under test.
Every cash frame is checked against the ledger's rules and the id contract (natural
events keep the world's ids; a refund's id is minus the payment it returns).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core import actions, ledger, world
from core.asof import PolicyState

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_world"
OBSERVED_UNTIL = pd.Timestamp("2025-06-30 23:59:59")
T = pd.Timestamp
ORDER_TABLES = (
    "plans", "installment_schedule", "payment_attempts", "payment_reversals", "fulfilments",
    "deliveries", "dispute_openings", "dispute_resolutions", "victim_reports", "plan_writeoffs",
)
UNTOUCHED = {
    "hold_at": pd.NaT, "hold_before_shipment": False, "released_at": pd.NaT,
    "hold_outcome": None, "hold_ended_at": pd.NaT, "void_at": pd.NaT, "void_cause": None,
}
NO_BLOCKS = pd.DataFrame({"user_id": pd.Series(dtype="int64"),
                          "at": pd.Series(dtype="datetime64[s]")})


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return world.read_world(FIXTURE)


@pytest.fixture(scope="module")
def terms() -> ledger.ProductTerms:
    return ledger.ProductTerms.from_config()


# ------------------------------------------------------------------ helpers


def _fates(tables, changes=None) -> pd.DataFrame:
    """Every processor-approved attempt approved at checkout, with per-order changes."""
    attempts = tables["order_attempts"]
    approved = attempts.loc[attempts["processor_result"] == "approved"]
    rows = []
    for order, user, checkout in approved[["order_id", "user_id", "known_at"]].itertuples(
            index=False):
        row = {"order_id": order, "user_id": user, "checkout_at": checkout,
               "route": "approve", **UNTOUCHED}
        row.update((changes or {}).get(order, {}))
        rows.append(row)
    return actions.typed_fates(pd.DataFrame(rows))


def _hold(at, *, before=True, outcome=None, release=pd.NaT, void=pd.NaT, cause=None,
          ended=None):
    """A hold; it ends at its release or void, or (after shipment) at ``ended`` or at once."""
    if outcome is None:
        end = pd.NaT
    elif not pd.isna(T(release)):
        end = T(release)
    elif not pd.isna(T(void)):
        end = T(void)
    else:
        end = T(ended if ended is not None else at)
    return {"route": "review", "hold_at": T(at), "hold_before_shipment": before,
            "hold_outcome": outcome, "released_at": T(release), "hold_ended_at": end,
            "void_at": T(void), "void_cause": cause}


def _order_of(tables, name, frame) -> pd.Series:
    """The order each row of ``frame`` (world or realized table ``name``) belongs to."""
    if "order_id" in frame.columns:
        return frame["order_id"]
    if name == "dispute_resolutions":
        return frame["dispute_id"].map(
            tables["dispute_openings"].set_index("dispute_id")["order_id"])
    return frame["plan_id"].map(tables["plans"].set_index("plan_id")["order_id"])


def _rows(tables, name, frame, orders, *, keep=True) -> pd.DataFrame:
    mine = _order_of(tables, name, frame).isin(list(orders))
    return frame.loc[mine if keep else ~mine].reset_index(drop=True)


def _plain(value):
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else value


def _records(frame, columns) -> list[tuple]:
    return [tuple(_plain(v) for v in row)
            for row in frame[list(columns)].itertuples(index=False, name=None)]


def _cash(frame) -> list[tuple]:
    """Cash events as a sorted list of plain tuples: a multiset blind to dtypes."""
    return sorted(
        (int(e), int(o), int(p), int(m), str(k), int(a), T(oc), T(kn), int(r), str(c))
        for e, o, p, m, k, a, oc, kn, r, c in frame[list(ledger.CASH_COLUMNS)].itertuples(
            index=False, name=None))


def _checked(tables, cash) -> pd.DataFrame:
    ledger.validate_cash_events(cash)
    assert cash["event_id"].notna().all()
    known = tables["cash_events"].set_index("event_id")
    natural = cash.loc[cash["cause"] == "natural"]
    found = known.loc[natural["event_id"].astype("int64")]
    assert found["kind"].tolist() == natural["kind"].tolist()
    assert found["ref_event_id"].tolist() == natural["ref_event_id"].tolist()
    acted = cash.loc[cash["cause"] != "natural"]
    assert acted["kind"].eq("refund").all()
    assert (acted["event_id"] == -acted["ref_event_id"]).all()
    return cash


def _net(cash, order) -> int:
    return int(cash.loc[cash["order_id"] == order, "amount_cents"].sum())


def _assert_others_unchanged(tables, realized, changed) -> None:
    """Every row outside the ``changed`` orders is the world's, row for row."""
    assert set(realized) == set(tables)
    for name, frame in tables.items():
        if name == "cash_events":
            mine = realized[name]
            assert _cash(mine.loc[~mine["order_id"].isin(list(changed))]) == _cash(
                frame.loc[~frame["order_id"].isin(list(changed))])
        elif name in ORDER_TABLES:
            pd.testing.assert_frame_equal(
                _rows(tables, name, realized[name], changed, keep=False),
                _rows(tables, name, frame, changed, keep=False), obj=name)
        else:
            pd.testing.assert_frame_equal(realized[name], frame, obj=name)


def _approved(state) -> dict:
    return dict(_records(state.approved, ["order_id", "approved_at"]))


# ------------------------------------------------------------------ approve-all


def test_approve_all_fates_reproduce_the_world(tables, terms):
    fates = actions.approve_all_fates(tables["order_attempts"])
    # 26 attempts less the two processor declines (orders 16 and 17), each at its checkout.
    assert list(fates.columns) == list(actions.FATE_COLUMNS)
    assert fates["order_id"].tolist() == [*range(1, 16), *range(18, 27)]
    pd.testing.assert_frame_equal(fates, _fates(tables))
    assert _records(fates.loc[fates["order_id"] == 18], ["user_id", "checkout_at", "route"]) == [
        (12, T("2025-02-14 01:40"), "approve")]

    cash = _checked(tables, actions.policy_cash(tables, fates, terms))
    assert _cash(cash) == _cash(tables["cash_events"])
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [])
    _checked(tables, realized["cash_events"])


# ------------------------------------------------------------------ declined at checkout


def test_declined_at_checkout_keeps_only_the_attempt(tables, terms):
    # dan's $899 takeover order 12 is auto-declined; eve's $650 order 18 meets a block.
    fates = _fates(tables, {12: {"route": "auto_decline"}, 18: {"route": "blocked"}})
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [12, 18])  # the attempts included
    for name in ORDER_TABLES:
        had = not _rows(tables, name, tables[name], [12, 18]).empty
        assert had == (name != "payment_reversals"), name  # the world had rows to remove
        assert _rows(tables, name, realized[name], [12, 18]).empty, name
    cash = _checked(tables, realized["cash_events"])
    assert not cash["order_id"].isin([12, 18]).any()
    # The world nets -49,445 on order 12 and -53,500 on order 18 (README): both are gone.
    assert int(cash["amount_cents"].sum()) == int(
        tables["cash_events"]["amount_cents"].sum()) + 49_445 + 53_500


# ------------------------------------------------------------------ declines and voids


@pytest.mark.parametrize("cause", ["decline", "escalate"])
def test_fraud_voided_before_shipping_nets_zero(tables, terms, cause):
    # eve's stolen-card order 18: checkout 2025-02-14 01:40 collects 25% of 65,000 = 16,250;
    # the world ships it at 09:40. The reviewer voids it at 05:00.
    void = T("2025-02-14 05:00")
    fates = _fates(tables, {18: {"route": "review", "void_at": void, "void_cause": cause}})
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [18])
    assert _rows(tables, "plans", realized["plans"], [18])["plan_id"].tolist() == [16]
    schedule = _rows(tables, "installment_schedule", realized["installment_schedule"], [18])
    assert _records(schedule, ["seq", "due_at", "amount_cents"]) == [
        (0, T("2025-02-14 01:40"), 16_250)]
    payments = _rows(tables, "payment_attempts", realized["payment_attempts"], [18])
    assert payments["event_id"].tolist() == [153]
    for name in ("fulfilments", "deliveries", "dispute_openings", "dispute_resolutions",
                 "plan_writeoffs"):
        assert _rows(tables, name, realized[name], [18]).empty, name

    cash = _checked(tables, realized["cash_events"])
    checkout = T("2025-02-14 01:40")
    assert _cash(cash.loc[cash["order_id"] == 18]) == sorted([
        (154, 18, 16, 2, "customer_payment", 16_250, checkout, checkout, 153, "natural"),
        (-153, 18, 16, 2, "refund", -16_250, void, void, 153, cause),
    ])
    # World: +16,250 down - 61,750 settlement (65,000 less the 3,250 fee) - 16,250 dispute
    # - 1,500 fee + 9,750 recovery (20% of the 48,750 written off) = -53,500.
    assert _net(tables["cash_events"], 18) == -53_500
    assert _net(cash, 18) == 0
    # The order went through at checkout (its down payment moved) and was voided at 05:00.
    state = actions.policy_state(fates, NO_BLOCKS)
    assert _approved(state)[18] == checkout
    assert _records(state.voided, ["order_id", "at"]) == [(18, void)]


def test_a_decline_one_minute_after_shipping_leaves_the_loss(tables, terms):
    # Order 18 shipped (and settled) at 09:40. A decline at 09:41 voids nothing: the order
    # keeps its route and every flow, and the block is the decline's only effect.
    fates = _fates(tables, {18: {"route": "review"}})
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [])
    assert _net(_checked(tables, realized["cash_events"]), 18) == -53_500
    blocks = pd.DataFrame({"user_id": [12], "at": [T("2025-02-14 09:41")]})
    state = actions.policy_state(fates, blocks)
    assert _approved(state)[18] == T("2025-02-14 01:40")
    assert state.voided.empty
    assert _records(state.blocked, ["user_id", "at"]) == [(12, T("2025-02-14 09:41"))]
    # A void after the settlement is refused rather than refunded.
    late = _fates(tables, {18: {"route": "review", "void_at": T("2025-02-14 09:41"),
                                "void_cause": "decline"}})
    with pytest.raises(ValueError, match="shipped"):
        actions.policy_cash(tables, late, terms)


def _with_checkout_reversal(tables, cents):
    """The mini world with a bank return of ``cents`` of order 18's checkout payment
    (event 153, plan 16, merchant 2) at 2025-02-14 03:00."""
    at = T("2025-02-14 03:00")
    out = dict(tables)
    reversals = tables["payment_reversals"]
    row = pd.DataFrame({"event_id": [1001], "occurred_at": [at], "known_at": [at],
                        "payment_event_id": [153], "plan_id": [16], "amount_cents": [cents],
                        "reason": ["bank_return"]})
    out["payment_reversals"] = pd.concat([reversals, row.astype(reversals.dtypes.to_dict())],
                                         ignore_index=True)
    cash = tables["cash_events"]
    row = pd.DataFrame({"event_id": [1002], "occurred_at": [at], "known_at": [at],
                        "order_id": [18], "plan_id": [16], "merchant_id": [2],
                        "kind": ["payment_reversal"], "amount_cents": [-cents],
                        "ref_event_id": [1001], "cause": ["natural"]})
    out["cash_events"] = pd.concat([cash, row.astype(cash.dtypes.to_dict())], ignore_index=True)
    return out


@pytest.mark.parametrize("reversed_cents, refund_cents", [(6_250, 10_000), (16_250, 0)])
def test_a_void_refunds_only_what_still_stands(tables, terms, reversed_cents, refund_cents):
    # Order 18's 16,250 checkout payment is returned by the bank (in part or whole) at 03:00;
    # the void at 05:00 refunds 16,250 - the returned amount, and nothing when none stands.
    synthetic = _with_checkout_reversal(tables, reversed_cents)
    void, checkout, returned = T("2025-02-14 05:00"), T("2025-02-14 01:40"), T("2025-02-14 03:00")
    fates = _fates(synthetic, {18: {"route": "review", "void_at": void, "void_cause": "decline"}})
    cash = _checked(synthetic, actions.policy_cash(synthetic, fates, terms))
    expected = [
        (154, 18, 16, 2, "customer_payment", 16_250, checkout, checkout, 153, "natural"),
        (1002, 18, 16, 2, "payment_reversal", -reversed_cents, returned, returned, 1001,
         "natural"),
    ]
    if refund_cents:
        expected.append((-153, 18, 16, 2, "refund", -refund_cents, void, void, 153, "decline"))
    assert _cash(cash.loc[cash["order_id"] == 18]) == sorted(expected)
    assert _net(cash, 18) == 0


# ------------------------------------------------------------------ holds before shipment


def test_a_released_hold_moves_shipment_by_the_pause_and_the_schedule_to_the_release(
        tables, terms):
    # gus's repaid $120 order 1: checkout 2024-12-05 12:00, world shipment 12-06 00:00.
    # Held at 18:00, verified at 12-07 10:00 (40 h later, inside 48 h): the shipment moves by
    # 40 h; installments 1-3 fall due 14, 28 and 42 days after the release (checkout + 46 h).
    fates = _fates(tables, {1: _hold("2024-12-05 18:00", outcome="cleared",
                                     release="2024-12-07 10:00")})
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [1])

    def times(name):
        return _records(_rows(tables, name, realized[name], [1]),
                        ["event_id", "occurred_at", "known_at"])

    shipped, delivered = T("2024-12-07 16:00"), T("2024-12-09 16:00")
    assert times("fulfilments") == [(4, shipped, shipped)]
    assert times("deliveries") == [(6, delivered, delivered)]
    due = [T("2024-12-05 12:00"), T("2024-12-21 10:00"), T("2025-01-04 10:00"),
           T("2025-01-18 10:00")]
    schedule = _rows(tables, "installment_schedule", realized["installment_schedule"], [1])
    assert _records(schedule, ["seq", "due_at", "amount_cents"]) == [
        (seq, at, 3_000) for seq, at in enumerate(due)]
    assert times("payment_attempts") == [
        (ev, at, at) for ev, at in zip([2, 13, 24, 64], due, strict=True)]

    # 4 x 3,000 collected less the settlement of 12,000 - 600 fee: +600, as in the world.
    cash = _checked(tables, realized["cash_events"])
    assert _cash(cash.loc[cash["order_id"] == 1]) == sorted([
        (3, 1, 1, 1, "customer_payment", 3_000, due[0], due[0], 2, "natural"),
        (5, 1, 1, 1, "merchant_settlement", -11_400, shipped, shipped, 4, "natural"),
        (14, 1, 1, 1, "customer_payment", 3_000, due[1], due[1], 13, "natural"),
        (25, 1, 1, 1, "customer_payment", 3_000, due[2], due[2], 24, "natural"),
        (65, 1, 1, 1, "customer_payment", 3_000, due[3], due[3], 64, "natural"),
    ])
    assert _net(cash, 1) == _net(tables["cash_events"], 1) == 600
    assert _approved(actions.policy_state(fates, NO_BLOCKS))[1] == T("2024-12-07 10:00")


# Released pre-shipment holds and the moves they imply (shipment clock = release - hold,
# payment clock = release - checkout):
#   order 12, checkout 02-03 02:20, held 03:20, released 02-04 03:20: +24 h and +25 h;
#   order 14, checkout 02-10 20:05, held 02-11 10:05, released 02-12 06:05: +20 h and +34 h;
#   order 18, checkout 02-14 01:40, held 05:00, released 02-15 20:00: +39 h and +42 h 20 min.
RELEASED = {
    12: _hold("2025-02-03 03:20", outcome="cleared", release="2025-02-04 03:20"),
    14: _hold("2025-02-11 10:05", outcome="cleared", release="2025-02-12 06:05"),
    18: _hold("2025-02-14 05:00", outcome="cleared", release="2025-02-15 20:00"),
}
MOVED_ROWS = [  # table, event id, occurred_at, known_at after the move
    ("fulfilments", 110, "2025-02-04 08:20", "2025-02-04 08:20"),  # 02-03 08:20 + 24 h
    ("deliveries", 120, "2025-02-06 08:20", "2025-02-06 08:20"),  # 02-05 08:20 + 24 h
    ("victim_reports", 130, "2025-02-10 10:00", "2025-02-10 10:00"),  # 02-09 10:00 + 24 h
    ("payment_attempts", 108, "2025-02-03 02:20", "2025-02-03 02:20"),  # checkout: stays
    ("payment_attempts", 161, "2025-02-18 03:20", "2025-02-18 03:20"),  # 02-17 02:20 + 25 h
    ("plan_writeoffs", 291, "2025-04-01 03:20", "2025-04-01 03:20"),  # 03-31 02:20 + 25 h
    ("fulfilments", 136, "2025-02-12 12:05", "2025-02-12 12:05"),  # 02-11 16:05 + 20 h
    ("payment_attempts", 134, "2025-02-10 20:05", "2025-02-10 20:05"),  # checkout: stays
    ("payment_attempts", 175, "2025-02-26 06:05", "2025-02-26 06:05"),  # 02-24 20:05 + 34 h
    ("payment_reversals", 182, "2025-02-28 06:05", "2025-02-28 06:05"),  # 02-26 20:05 + 34 h
    ("payment_attempts", 185, "2025-03-02 06:05", "2025-03-02 06:05"),  # 02-28 20:05 + 34 h
    ("fulfilments", 155, "2025-02-16 00:40", "2025-02-16 00:40"),  # 02-14 09:40 + 39 h
    ("deliveries", 158, "2025-02-18 00:40", "2025-02-18 00:40"),  # 02-16 09:40 + 39 h
    ("dispute_openings", 197, "2025-03-03 00:00", "2025-03-05 00:00"),  # 03-01 09:00, 03-03 09:00
    ("dispute_resolutions", 306, "2025-04-10 00:00", "2025-04-12 00:00"),  # 04-08, 04-10 09:00
    ("payment_attempts", 153, "2025-02-14 01:40", "2025-02-14 01:40"),  # checkout: stays
    ("payment_attempts", 184, "2025-03-01 20:00", "2025-03-01 20:00"),  # 02-28 01:40 + 42:20
    ("plan_writeoffs", 307, "2025-04-12 20:00", "2025-04-12 20:00"),  # 04-11 01:40 + 42:20
]
MOVED_CASH = [  # event id, amount, time after the move (occurred and known)
    (109, 22_475, "2025-02-03 02:20"),  # 25% of 89,900 at checkout
    (111, -85_405, "2025-02-04 08:20"),  # 89,900 - 4,495 fee, at the moved shipment
    (325, 13_485, "2025-05-01 03:20"),  # 20% of 67,425, 30 days after the moved write-off
    (135, 10_750, "2025-02-10 20:05"),
    (137, -40_850, "2025-02-12 12:05"),  # 43,000 - 2,150 fee
    (183, -10_750, "2025-02-28 06:05"),  # the bank return, with its payment
    (154, 16_250, "2025-02-14 01:40"),
    (156, -61_750, "2025-02-16 00:40"),
    (198, -16_250, "2025-03-05 00:00"),  # dispute debit when the moved opening is known
    (199, -1_500, "2025-03-05 00:00"),  # dispute fee
    (329, 9_750, "2025-05-12 20:00"),  # 20% of 48,750, 30 days after 04-12 20:00
]


def test_a_released_hold_moves_disputes_reports_reversals_and_write_offs(tables, terms):
    fates = _fates(tables, RELEASED)
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, RELEASED)
    for name in ORDER_TABLES:  # nothing crosses the observation end: no row is lost
        assert len(_rows(tables, name, realized[name], RELEASED)) == len(
            _rows(tables, name, tables[name], RELEASED)), name
    for name, event, occurred, known in MOVED_ROWS:
        row = realized[name].loc[realized[name]["event_id"] == event]
        assert _records(row, ["occurred_at", "known_at"]) == [(T(occurred), T(known))], event
    schedule = realized["installment_schedule"]
    plan_14 = schedule.loc[schedule["plan_id"] == 14]
    assert plan_14["due_at"].tolist() == [  # release 02-12 06:05 + 0, 14, 28, 42 days
        T("2025-02-10 20:05"), T("2025-02-26 06:05"), T("2025-03-12 06:05"),
        T("2025-03-26 06:05")]

    cash = _checked(tables, realized["cash_events"])
    for event, amount, at in MOVED_CASH:
        row = cash.loc[cash["event_id"] == event]
        assert _records(row, ["amount_cents", "occurred_at", "known_at"]) == [
            (amount, T(at), T(at))], event
    ignoring_time = ["event_id", "kind", "amount_cents", "ref_event_id", "cause"]
    for order, net in ((12, -49_445), (14, 2_150), (18, -53_500)):
        mine, theirs = (frame.loc[frame["order_id"] == order]
                        for frame in (cash, tables["cash_events"]))
        assert sorted(_records(mine, ignoring_time)) == sorted(_records(theirs, ignoring_time))
        assert _net(cash, order) == net
    approved = _approved(actions.policy_state(fates, NO_BLOCKS))
    assert [approved[order] for order in RELEASED] == [
        T("2025-02-04 03:20"), T("2025-02-12 06:05"), T("2025-02-15 20:00")]


def test_events_a_release_moves_past_the_observation_end_are_dropped(tables, terms):
    # mo's $30 order 26: checkout 2025-06-10 10:00, world shipment 22:00, installment 1 paid
    # 06-24 10:00 (the world's last event). Held at 12:00, released 06-12 08:00: shipment
    # +44 h to 06-12 18:00, installment 1 +46 h to 06-26 08:00, past an end at 06-25 00:00.
    fates = _fates(tables, {26: _hold("2025-06-10 12:00", outcome="cleared",
                                      release="2025-06-12 08:00")})
    end = T("2025-06-25 00:00")
    realized = actions.realize(tables, fates, terms, observed_until=end)
    _assert_others_unchanged(tables, realized, [26])
    payments = _rows(tables, "payment_attempts", realized["payment_attempts"], [26])
    assert payments["event_id"].tolist() == [334]
    shipped = T("2025-06-12 18:00")
    assert _records(_rows(tables, "fulfilments", realized["fulfilments"], [26]),
                    ["event_id", "occurred_at"]) == [(336, shipped)]
    cash = _checked(tables, realized["cash_events"])
    checkout = T("2025-06-10 10:00")
    assert _cash(cash.loc[cash["order_id"] == 26]) == sorted([
        (335, 26, 24, 1, "customer_payment", 750, checkout, checkout, 334, "natural"),
        (337, 26, 24, 1, "merchant_settlement", -2_850, shipped, shipped, 336, "natural"),
    ])
    assert _net(cash, 26) == 750 - 2_850  # -2,100; the world has -1,350 with installment 1
    # Without an observation end the moved payment stays.
    unbounded = _checked(tables, actions.policy_cash(tables, fates, terms))
    assert _net(unbounded, 26) == _net(tables["cash_events"], 26) == -1_350


def test_cash_a_release_moves_past_the_observation_end_is_dropped(tables, terms):
    fates = _fates(tables, RELEASED)
    # dan's recovery (20% of the write-off, 30 days after it) moves to 05-01 03:20 and
    # eve's to 05-12 20:00: past an end at 05-01 00:00, though their write-offs are not
    end = T("2025-05-01 00:00")
    cash = _checked(tables, actions.policy_cash(tables, fates, terms, observed_until=end))
    assert not cash["event_id"].isin([325, 329]).any()
    mine = cash.loc[cash["order_id"].isin(list(RELEASED))]
    assert (mine["known_at"] <= end).all()
    assert _net(cash, 12) == -49_445 - 13_485
    # eve's dispute opening moves to 03-03 00:00 but is known only 03-05 00:00: with an
    # end at 03-04 12:00 it is not observed, and neither is its cash
    end = T("2025-03-04 12:00")
    realized = actions.realize(tables, fates, terms, observed_until=end)
    assert 197 not in set(realized["dispute_openings"]["event_id"])
    assert not realized["cash_events"]["event_id"].isin([198, 199]).any()
    for name in ORDER_TABLES:  # every moved observation of the three orders is known by then
        frame = _rows(tables, name, realized[name], RELEASED)
        if "known_at" in frame:
            assert (frame["known_at"] <= end).all(), name


@pytest.mark.parametrize("order, hold, outcome, void, cause, down, plan, merchant, payment", [
    # lee's $155 order 19 (checkout 03-02 13:00, world shipment 03-03 01:00, inside the hold):
    # no response, cancelled at hold + 48 h; 25% of 15,500 = 3,875 refunded.
    (19, "2025-03-02 14:00", "cancelled", "2025-03-04 14:00", "hold_cancelled", 3_875, 17, 1,
     191),
    # dan's $899 order 12 (checkout 02-03 02:20, world shipment 08:20): the check fails at
    # 09:00, after the world's shipment time, but the hold paused it; 22,475 refunded.
    (12, "2025-02-03 03:20", "declined", "2025-02-03 09:00", "decline", 22_475, 12, 2, 108),
])
def test_a_hold_that_ends_without_release_keeps_only_what_happened_by_the_hold(
        tables, terms, order, hold, outcome, void, cause, down, plan, merchant, payment):
    fates = _fates(tables, {order: _hold(hold, outcome=outcome, void=void, cause=cause)})
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [order])
    checkout = tables["order_attempts"].set_index("order_id").at[order, "known_at"]
    schedule = _rows(tables, "installment_schedule", realized["installment_schedule"], [order])
    assert _records(schedule, ["seq", "due_at", "amount_cents"]) == [(0, checkout, down)]
    assert _rows(tables, "payment_attempts", realized["payment_attempts"], [order])[
        "event_id"].tolist() == [payment]
    for name in ("fulfilments", "deliveries", "dispute_openings", "victim_reports",
                 "plan_writeoffs"):
        assert _rows(tables, name, realized[name], [order]).empty, name
    cash = _checked(tables, realized["cash_events"])
    assert _cash(cash.loc[cash["order_id"] == order]) == sorted([  # cash id = payment id + 1
        (payment + 1, order, plan, merchant, "customer_payment", down, checkout, checkout,
         payment, "natural"),
        (-payment, order, plan, merchant, "refund", -down, T(void), T(void), payment, cause),
    ])
    assert _net(cash, order) == 0  # the world: +775 on order 19, -49,445 on order 12

    state = actions.policy_state(fates, NO_BLOCKS)
    assert order not in _approved(state)
    assert _records(state.held, ["order_id", "held_at", "released_at", "outcome",
                                 "before_shipment"]) == [(order, T(hold), None, outcome, True)]
    assert _records(state.voided, ["order_id", "at"]) == [(order, T(void))]


# ------------------------------------------------------------------ holds after shipment


def test_a_pending_hold_after_shipment_pauses_nothing(tables, terms):
    # gus's order 1 shipped 2024-12-06 00:00; checks start 12-10 and have not finished.
    fates = _fates(tables, {1: _hold("2024-12-10 00:00", before=False)})
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [])  # order 1's rows and cash included
    assert _net(_checked(tables, realized["cash_events"]), 1) == 600
    state = actions.policy_state(fates, NO_BLOCKS)
    assert _approved(state)[1] == T("2024-12-05 12:00")
    assert state.voided.empty


# ------------------------------------------------------------------ as of a moment


AS_OF = {
    # fay's $520 order 11 (checkout 02-01 12:30, ships 02-02 06:30) voided at 06:00.
    11: {"route": "review", "void_at": T("2025-02-02 06:00"), "void_cause": "decline"},
    # jay's order 15 (shipped 02-13 02:00): a hold after shipment, check failed.
    15: _hold("2025-02-20 10:00", before=False, outcome="declined"),
    # eve's order 18: held before shipment, no response, cancelled 48 h later.
    18: _hold("2025-02-14 05:00", outcome="cancelled", void="2025-02-16 05:00",
              cause="hold_cancelled"),
    # lee's order 19: held before shipment, released the next morning.
    19: _hold("2025-03-02 14:00", outcome="cleared", release="2025-03-03 10:00"),
}
NOT_YET = (None, False, None, None, None, None)


def _acted(fates) -> dict:
    columns = ["hold_at", "hold_before_shipment", "released_at", "hold_outcome", "void_at",
               "void_cause"]
    return {row[0]: row[1:] for row in _records(fates, ["order_id", *columns])
            if row[0] in AS_OF}


@pytest.mark.parametrize("at, last_order, expected", [
    ("2025-02-02 00:00", 11, {11: NOT_YET}),  # the void comes later
    ("2025-02-03 02:20", 11, {  # order 12 checks out exactly now: absent
        11: (None, False, None, None, T("2025-02-02 06:00"), "decline")}),
    ("2025-02-15 00:00", 18, {
        11: (None, False, None, None, T("2025-02-02 06:00"), "decline"),
        15: NOT_YET,  # held on 02-20
        18: (T("2025-02-14 05:00"), True, None, None, None, None),  # pending
    }),
    ("2025-02-20 10:00", 18, {  # the hold on order 15 is placed exactly now: not yet seen
        11: (None, False, None, None, T("2025-02-02 06:00"), "decline"),
        15: NOT_YET,
        18: (T("2025-02-14 05:00"), True, None, "cancelled", T("2025-02-16 05:00"),
             "hold_cancelled"),
    }),
    ("2025-03-03 00:00", 19, {
        11: (None, False, None, None, T("2025-02-02 06:00"), "decline"),
        15: (T("2025-02-20 10:00"), False, None, "declined", None, None),
        18: (T("2025-02-14 05:00"), True, None, "cancelled", T("2025-02-16 05:00"),
             "hold_cancelled"),
        19: (T("2025-03-02 14:00"), True, None, None, None, None),  # released at 10:00
    }),
])
def test_fates_as_of_show_only_what_was_decided_before(tables, at, last_order, expected):
    fates = actions.fates_as_of(_fates(tables, AS_OF), T(at))
    present = [*range(1, 16), *range(18, 27)]
    assert fates["order_id"].tolist() == [o for o in present if o <= last_order]
    assert _acted(fates) == expected
    rest = fates.loc[~fates["order_id"].isin(list(AS_OF))]
    assert rest["route"].eq("approve").all() and rest["hold_at"].isna().all()


def test_an_outcome_that_comes_later_is_pending_after_shipment_too(tables):
    # order 14 shipped 2025-02-11 16:05; held after shipment on 02-20 10:00, declined 02-21
    fates = _fates(tables, {14: _hold("2025-02-20 10:00", before=False, outcome="declined",
                                      ended="2025-02-21 09:00")})
    early = actions.fates_as_of(fates, T("2025-02-21 00:00")).set_index("order_id")
    assert early.loc[14, "hold_outcome"] is None and pd.isna(early.loc[14, "hold_ended_at"])
    late = actions.fates_as_of(fates, T("2025-02-21 09:00:01")).set_index("order_id")
    assert late.loc[14, "hold_outcome"] == "declined"


def test_a_pending_hold_keeps_only_what_happened_by_the_hold(tables, terms):
    # As of 2025-02-15 00:00, eve's order 18 has been held since 05:00 the day before (the
    # cancellation comes on 02-16) and fay's order 11 was voided at 02-02 06:00.
    fates = actions.fates_as_of(_fates(tables, AS_OF), T("2025-02-15 00:00"))
    realized = actions.realize(tables, fates, terms, observed_until=OBSERVED_UNTIL)
    _assert_others_unchanged(tables, realized, [11, 18])
    for order, payment in ((11, 97), (18, 153)):
        assert _rows(tables, "payment_attempts", realized["payment_attempts"], [order])[
            "event_id"].tolist() == [payment]
        assert _rows(tables, "fulfilments", realized["fulfilments"], [order]).empty
    cash = _checked(tables, realized["cash_events"])
    checkout, voided = T("2025-02-14 01:40"), T("2025-02-02 06:00")
    assert _cash(cash.loc[cash["order_id"] == 18]) == [
        (154, 18, 16, 2, "customer_payment", 16_250, checkout, checkout, 153, "natural")]
    assert _cash(cash.loc[cash["order_id"] == 11]) == sorted([  # 25% of 52,000, refunded
        (98, 11, 11, 2, "customer_payment", 13_000, T("2025-02-01 12:30"),
         T("2025-02-01 12:30"), 97, "natural"),
        (-97, 11, 11, 2, "refund", -13_000, voided, voided, 97, "decline"),
    ])

    state = actions.policy_state(fates, NO_BLOCKS)
    assert 18 not in _approved(state)  # held before shipment: not through yet
    assert _approved(state)[11] == T("2025-02-01 12:30")  # through at checkout, then voided
    assert _records(state.held, ["order_id", "held_at", "released_at", "outcome",
                                 "before_shipment"]) == [
        (18, T("2025-02-14 05:00"), None, None, True)]
    assert _records(state.voided, ["order_id", "at"]) == [(11, voided)]


# ------------------------------------------------------------------ policy state


def test_policy_state_records_approvals_holds_voids_and_first_blocks(tables):
    attempts = tables["order_attempts"]
    approved = attempts.loc[attempts["processor_result"] == "approved"]
    fates = _fates(tables, {
        2: {"route": "review"},
        12: {"route": "auto_decline"},
        13: {"route": "blocked"},
        14: _hold("2025-02-20 10:00", before=False, outcome="cleared"),
        15: _hold("2025-02-20 10:00", before=False, outcome="declined"),
        18: {"route": "review", "void_at": T("2025-02-14 05:00"), "void_cause": "decline"},
        19: _hold("2025-03-02 14:00", outcome="cleared", release="2025-03-03 10:00"),
    })
    blocks = pd.DataFrame({  # eve blocked twice (the later one listed first), jay once
        "user_id": [12, 9, 12],
        "at": [T("2025-03-01 00:00"), T("2025-02-21 10:00"), T("2025-02-14 05:00")]})
    state = actions.policy_state(fates, blocks)

    # Approved at checkout (approve, review, holds after shipment, and order 18, which went
    # through before its void), order 19 at its release; orders 12 and 13 never went through.
    expected = {order: at for order, at in approved[["order_id", "known_at"]].itertuples(
        index=False) if order not in (12, 13)}
    expected[19] = T("2025-03-03 10:00")
    assert _approved(state) == expected
    assert sorted(_records(state.held, ["order_id", "held_at", "released_at", "outcome",
                                        "before_shipment"])) == [
        (14, T("2025-02-20 10:00"), None, "cleared", False),
        (15, T("2025-02-20 10:00"), None, "declined", False),
        (19, T("2025-03-02 14:00"), T("2025-03-03 10:00"), "cleared", True),
    ]
    assert _records(state.voided, ["order_id", "at"]) == [(18, T("2025-02-14 05:00"))]
    assert sorted(_records(state.blocked, ["user_id", "at"])) == [
        (9, T("2025-02-21 10:00")), (12, T("2025-02-14 05:00"))]


def test_approve_all_state_matches_the_reference(tables):
    state = actions.policy_state(actions.approve_all_fates(tables["order_attempts"]), NO_BLOCKS)
    reference = PolicyState.approve_all(tables)
    assert _approved(state) == _approved(reference)
    assert len(_approved(state)) == 24
    assert state.held.empty and state.voided.empty and state.blocked.empty


# ------------------------------------------------------------------ consistency


def _edit(fates, order, **values):
    out = fates.copy()
    row = out.index[out["order_id"] == order][0]
    for column, value in values.items():
        out.at[row, column] = value
    return out


@pytest.mark.parametrize("change, message", [
    (lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True), "two fates"),
    (lambda f: _edit(f, 1, route="flag"), "unknown routes"),
    (lambda f: f.drop(columns="void_cause"), "lack columns"),
    # a release after a hold placed after shipment, after a cancelled hold, or with no hold
    (lambda f: _edit(f, 1, hold_at=T("2024-12-06 06:00"), hold_outcome="cleared",
                     released_at=T("2024-12-06 08:00")), "release"),
    (lambda f: _edit(f, 18, hold_at=T("2025-02-14 05:00"), hold_before_shipment=True,
                     hold_outcome="cancelled", released_at=T("2025-02-15 20:00")), "release"),
    (lambda f: _edit(f, 18, hold_before_shipment=True, hold_outcome="cleared",
                     released_at=T("2025-02-15 20:00")), "release"),
    (lambda f: _edit(f, 18, void_at=T("2025-02-14 01:00"), void_cause="decline"),
     "void before checkout"),
    (lambda f: _edit(f, 12, route="auto_decline", hold_at=T("2025-02-03 03:00"),
                     hold_before_shipment=True), "declined at checkout"),
    (lambda f: _edit(f, 18, void_at=T("2025-02-14 05:00")), "void_at and void_cause"),
], ids=["two-fates", "unknown-route", "missing-column", "release-after-shipment-hold",
        "release-after-cancelled-hold", "release-without-hold", "void-before-checkout",
        "declined-at-checkout-and-held", "void-without-cause"])
def test_typed_fates_rejects_inconsistent_fates(tables, change, message):
    with pytest.raises(ValueError, match=message):
        actions.typed_fates(change(_fates(tables)))
