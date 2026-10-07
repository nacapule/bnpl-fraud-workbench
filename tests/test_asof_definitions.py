"""Named definitions in the as-of context, each checked on a small hand-built world:
attempts versus approved orders, first attempts, address linkage windows, email
identity, rows that arrive later, home geography, the travel-speed boundary,
installments paid and the never-pay determination under a policy."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from core import asof, world
from rules.engine import fired, score

T0 = pd.Timestamp("2025-03-01 12:00:00")


class Builder:
    """A tiny world in the world schema: accounts with one device, card and home address
    each, and order attempts on them."""

    def __init__(self) -> None:
        self.accounts: list[dict] = []
        self.orders: list[dict] = []
        self.events: list[dict] = []
        self.extra_links: list[dict] = []
        self.plans: list[dict] = []
        self.schedule: list[dict] = []
        self.payments: list[dict] = []
        self.reports: list[dict] = []

    def account(self, user_id: int, email: str, created: pd.Timestamp, *, country: str = "US",
                address: int | None = None) -> None:
        self.accounts.append({"user_id": user_id, "email": email, "created_at": created,
                              "country": country, "address": address or user_id})

    def order(self, order_id: int, user_id: int, at: pd.Timestamp, *, amount: int = 10_000,
              result: str = "approved", ip_country: str | None = None, avs: str = "Y",
              merchant: int = 1, address: int | None = None, device: int | None = None) -> None:
        owner = next(a for a in self.accounts if a["user_id"] == user_id)
        self.orders.append({"order_id": order_id, "user_id": user_id, "at": at,
                            "amount": amount, "result": result, "avs": avs,
                            "ip_country": ip_country or owner["country"], "merchant": merchant,
                            "address": address or owner["address"],
                            "device": device or user_id})

    def plan(self, order_id: int, *, paid: tuple[int, ...] = ()) -> int:
        """A pay-in-4 plan on an approved order: 2,500 cents at checkout, then three
        installments of 2,500 every 14 days; the checkout payment and the ``paid``
        installments succeed on their due dates."""
        order = next(o for o in self.orders if o["order_id"] == order_id)
        plan_id = len(self.plans) + 1
        self.plans.append({"plan_id": plan_id, "order_id": order_id, "created_at": order["at"],
                           "principal_cents": 10_000, "down_payment_cents": 2_500,
                           "n_installments": 3})
        for seq in range(4):
            due = order["at"] + pd.Timedelta(days=14 * seq)
            self.schedule.append({"plan_id": plan_id, "seq": seq, "due_at": due,
                                  "amount_cents": 2_500})
            if seq == 0 or seq in paid:
                self.payment(plan_id, seq, due)
        return plan_id

    def payment(self, plan_id: int, seq: int, at: pd.Timestamp) -> None:
        self.payments.append({"event_id": len(self.payments) + 1, "occurred_at": at,
                              "known_at": at, "plan_id": plan_id, "seq": seq, "attempt_no": 1,
                              "amount_cents": 2_500, "result": "success"})

    def report(self, order_id: int, at: pd.Timestamp) -> None:
        """The account's owner disowns the order."""
        order = next(o for o in self.orders if o["order_id"] == order_id)
        self.reports.append({"event_id": len(self.reports) + 1, "occurred_at": at,
                             "known_at": at, "user_id": order["user_id"],
                             "order_id": order_id})

    def tables(self) -> dict[str, pd.DataFrame]:
        accounts = pd.DataFrame(self.accounts)
        start = pd.Timestamp("2020-01-01")
        orders = pd.DataFrame(self.orders)
        users = accounts["user_id"]
        addresses = sorted(set(accounts["address"]) | set(orders["address"]))
        devices = sorted(set(users) | set(orders["device"]))
        frames = {
            "accounts": pd.DataFrame({
                "user_id": users, "created_at": accounts["created_at"],
                "email": accounts["email"], "email_domain": accounts["email"].str.split("@").str[1],
                "home_country": accounts["country"], "dob_year": 1990}),
            "merchants": pd.DataFrame({
                "merchant_id": [1, 2], "created_at": start, "name": ["m1", "m2"],
                "category": ["electronics", "jewelry"], "risk_tier": 1,
                "fulfilment_median_hours": 12.0, "closed_at": pd.NaT}),
            "devices": pd.DataFrame({"device_id": devices, "created_at": start,
                                     "fingerprint": "f", "ua_family": "x"}),
            "device_links": pd.DataFrame(
                [{"user_id": u, "device_id": u, "created_at": c, "removed_at": pd.NaT}
                 for u, c in zip(users, accounts["created_at"], strict=True)]
                + [{"user_id": o["user_id"], "device_id": o["device"],
                    "created_at": start, "removed_at": pd.NaT}
                   for o in self.orders if o["device"] != o["user_id"]]).drop_duplicates(
                ["user_id", "device_id"]),
            "addresses": pd.DataFrame({"address_id": addresses, "created_at": start,
                                       "line_hash": "h", "city": "c", "region": "r",
                                       "country": "US"}),
            "address_links": pd.DataFrame(
                [{"user_id": a["user_id"], "address_id": a["address"],
                  "created_at": a["created_at"], "removed_at": pd.NaT, "role": "home"}
                 for a in self.accounts] + self.extra_links),
            "cards": pd.DataFrame({"card_id": users, "user_id": users,
                                   "created_at": accounts["created_at"], "removed_at": pd.NaT,
                                   "bin_country": accounts["country"], "network": "visa",
                                   "last4": "0000"}),
            "order_attempts": pd.DataFrame({
                "event_id": range(1, len(orders) + 1), "occurred_at": orders["at"],
                "known_at": orders["at"], "order_id": orders["order_id"],
                "user_id": orders["user_id"], "merchant_id": orders["merchant"],
                "device_id": orders["device"], "card_id": orders["user_id"],
                "ship_address_id": orders["address"], "amount_cents": orders["amount"],
                "promo_id": pd.array([pd.NA] * len(orders), dtype="Int64"),
                "promo_discount_cents": 0, "ip": "1.1.1.1", "ip_country": orders["ip_country"],
                "avs_result": orders["avs"], "cvv_result": "M",
                "processor_result": orders["result"]}),
            "account_events": pd.DataFrame(self.events) if self.events
            else world.empty("account_events"),
        }
        for name, rows in (("plans", self.plans), ("installment_schedule", self.schedule),
                           ("payment_attempts", self.payments),
                           ("victim_reports", self.reports)):
            if rows:
                frames[name] = pd.DataFrame(rows)
        tables = {name: world.coerce(name, frame) for name, frame in frames.items()}
        for name in asof.INPUT_TABLES:
            tables.setdefault(name, world.empty(name))
        return tables


def context_of(builder: Builder, decisions: pd.DataFrame | None = None) -> pd.DataFrame:
    return asof.build_context(builder.tables(), decisions).set_index("order_id")


def household() -> Builder:
    b = Builder()
    for user in (1, 2, 3):
        b.account(user, f"user{user}@outlook.com", T0 - pd.Timedelta(days=400))
    return b


def test_attempts_and_approved_orders_are_separate_columns() -> None:
    """Two processor declines then an approval: three attempts in 24 hours, no earlier
    approved order, and the approval is not the account's first attempt."""
    b = household()
    for k, result in enumerate(("declined", "declined", "approved")):
        b.order(10 + k, 1, T0 + pd.Timedelta(minutes=10 * k), result=result)
    row = context_of(b).loc[12]
    assert row["attempts_user_24h"] == 3
    assert row["approved_orders_user_24h"] == 0
    assert row["is_first_attempt_user"] == 0
    assert row["processor_declines_card_24h"] == 2


def test_address_linkage_in_30_days_differs_from_ever() -> None:
    """A housemate registered the address long ago and ordered to it 40 days before: the
    30-day count sees only recent attempts, the lifetime count sees both accounts."""
    b = household()
    b.accounts[1]["address"] = 1
    b.order(20, 2, T0 - pd.Timedelta(days=40))
    b.order(21, 1, T0)
    row = context_of(b).loc[21]
    assert row["accounts_on_address_30d"] == 1
    assert row["accounts_on_address_ever"] == 2


@pytest.mark.parametrize(("first", "second", "linked"), [
    ("alice@gmail.com", "alice+1@gmail.com", 1),  # plus tags removed for every provider
    ("j.doe@outlook.com", "jdoe@outlook.com", 0),  # dots kept outside Gmail
    ("a.b@googlemail.com", "ab@gmail.com", 1),  # googlemail is Gmail
    ("ab12@gmail.com", "ab@gmail.com", 0),  # digits are never stripped
])
def test_email_identity_in_the_context(first: str, second: str, linked: int) -> None:
    b = Builder()
    b.account(1, first, T0 - pd.Timedelta(days=10))
    b.account(2, second, T0 - pd.Timedelta(days=5))
    b.order(30, 1, T0)
    assert context_of(b).loc[30, "email_root_other_accounts"] == linked


def test_email_root_count_is_other_holders_at_the_decision() -> None:
    """Three other accounts hold the address before the order and one more signs up after
    it: the order sees exactly three, neither the later account nor itself."""
    b = Builder()
    for user, email in enumerate(("ann@gmail.com", "a.nn@gmail.com", "ann+x@gmail.com",
                                  "ann+y@gmail.com"), start=1):
        b.account(user, email, T0 - pd.Timedelta(days=10 * user))
    b.account(5, "an.n@gmail.com", T0 + pd.Timedelta(hours=1))
    b.order(40, 1, T0)
    assert context_of(b).loc[40, "email_root_other_accounts"] == 3


def test_rows_that_arrive_later_change_no_earlier_row_or_score() -> None:
    """A large approval in the category and a look-alike account, both after the order,
    leave its context row and rule score unchanged; so does reordering input rows."""
    b = Builder()
    for user in range(1, 121):
        b.account(user, f"u{user}@outlook.com", T0 - pd.Timedelta(days=30))
        b.order(1000 + user, user, T0 - pd.Timedelta(days=20) + pd.Timedelta(minutes=user),
                amount=10_000 + 100 * user)
    b.account(500, "new@outlook.com", T0 - pd.Timedelta(days=2))
    b.order(50, 500, T0, amount=30_000)
    before = context_of(b)
    assert before.loc[50, "amount_over_category_p95"] > 1  # R04 fires on the first order
    b.account(501, "new+1@outlook.com", T0 + pd.Timedelta(days=300))
    b.order(51, 501, T0 + pd.Timedelta(days=300), amount=900_000)
    after = context_of(b)
    pd.testing.assert_frame_equal(before, after.drop(index=51))
    assert score(before.loc[[50]])[0] == score(after.loc[[50]])[0]
    shuffled = {name: frame.sample(frac=1, random_state=3) for name, frame in b.tables().items()}
    pd.testing.assert_frame_equal(after, asof.build_context(shuffled).set_index("order_id")
                                  .loc[after.index])


def test_simultaneous_attempts_follow_the_tie_order() -> None:
    """Attempts in the same second count only those ordered before them (by event id)."""
    b = household()
    for order_id in (60, 61, 62):
        b.order(order_id, 1, T0)
    counts = context_of(b)["attempts_user_24h"]
    assert counts.loc[[60, 61, 62]].tolist() == [1, 2, 3]


def test_home_geography_comes_from_the_account() -> None:
    """A Canadian customer at home with a Canadian card is not abroad, and an AVS failure
    alone does not make R03 fire; the same customer ordering from France is abroad."""
    b = Builder()
    b.account(1, "c@outlook.com", T0 - pd.Timedelta(days=300), country="CA")
    b.order(70, 1, T0, avs="N")
    b.order(71, 1, T0 + pd.Timedelta(days=3), ip_country="FR", avs="N")
    ctx = context_of(b)
    assert ctx.loc[70, "ip_country_not_home"] == 0 and ctx.loc[71, "ip_country_not_home"] == 1
    assert ctx.loc[70, "bin_ip_country_mismatch"] == 0
    flags = fired(ctx)
    assert not flags.loc[70, "R03"] and flags.loc[71, "R03"]


@pytest.mark.parametrize(("gap", "fires"), [("11:59:59", True), ("12:00:00", False)])
def test_travel_speed_counts_only_attempts_under_12_hours_apart(gap: str, fires: bool) -> None:
    b = Builder()
    b.account(1, "t@outlook.com", T0 - pd.Timedelta(days=300))
    b.order(80, 1, T0)
    b.order(81, 1, T0 + pd.Timedelta(gap), ip_country="ID")  # about 15,000 km away
    row = context_of(b).loc[81]
    assert bool(row["geo_kmh_from_previous_attempt"] > 900) is fires
    assert bool(fired(context_of(b).loc[[81]])["R11"].iloc[0]) is fires


def test_a_review_later_sees_linkage_known_by_then() -> None:
    """Linkage is counted at the decision: accounts that use the device while the order
    waits count at review time, while order-anchored columns stay as at checkout."""
    b = household()
    b.order(90, 1, T0)
    b.order(91, 2, T0 + pd.Timedelta(hours=2), device=1)
    b.order(92, 3, T0 + pd.Timedelta(hours=3), device=1)
    review = pd.DataFrame({"order_id": [90, 90], "decision_at": [T0, T0 + pd.Timedelta(hours=4)]})
    rows = asof.build_context(b.tables(), review)
    assert rows["accounts_on_device_30d"].tolist() == [1, 3]
    assert rows["attempts_device_24h"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="before the order's checkout"):
        asof.build_context(b.tables(), review.assign(decision_at=T0 - pd.Timedelta(seconds=1)))


def test_installments_of_different_plans_are_never_confused() -> None:
    """Two plans opened a day apart with only their checkout payments: at day 14.5 the
    first plan's installment is due and unpaid, whatever the second plan's checkout
    payment, and a payment made later changes nothing."""
    b = household()
    b.order(100, 1, T0)
    b.order(101, 1, T0 + pd.Timedelta(days=1))
    second = None
    for order_id in (100, 101):
        second = b.plan(order_id)
    b.order(102, 1, T0 + pd.Timedelta(days=14, hours=12))
    before = context_of(b).loc[102]
    assert before["installments_due_user"] == 1
    assert before["installments_paid_user"] == 0
    b.payment(second, 3, T0 + pd.Timedelta(days=43))
    after = context_of(b).loc[102]
    pd.testing.assert_series_equal(before, after)


def zero_effort_household() -> Builder:
    """An account opens two plans a day apart and pays nothing after checkout: both
    default 30 days after their first installment (days 44 and 45), each the other's
    intent marker. Order 112 comes at day 60."""
    b = household()
    b.order(110, 1, T0)
    b.order(111, 1, T0 + pd.Timedelta(days=1))
    b.plan(110)
    b.plan(111)
    b.order(112, 1, T0 + pd.Timedelta(days=60))
    return b


def under_policy(b: Builder, voided_after: pd.Timedelta | None) -> pd.Series:
    """Order 112's outcome columns when the policy voids orders 110 and 111
    ``voided_after`` their checkouts (never when None)."""
    tables = b.tables()
    state = asof.PolicyState.approve_all(tables)
    if voided_after is not None:
        at = tables["order_attempts"].set_index("order_id").loc[[110, 111], "known_at"]
        state = dataclasses.replace(state, voided=pd.DataFrame(
            {"order_id": [110, 111], "at": (at + voided_after).to_numpy()}))
    decisions = pd.DataFrame({"order_id": [112], "decision_at": [T0 + pd.Timedelta(days=60)]})
    return asof.outcome_columns(tables, state, decisions).iloc[0]


def test_never_pay_is_determined_from_two_zero_effort_plans() -> None:
    row = under_policy(zero_effort_household(), None)
    assert row["installments_due_user"] == 6 and row["installments_paid_user"] == 0
    assert row["never_pay_determined_user"] == 1


@pytest.mark.parametrize(("voided_after", "determined"), [
    (pd.Timedelta(hours=1), 0),  # voided before shipment: nothing was owed
    (pd.Timedelta(days=50), 1),  # voided after both defaults: the determination stands
])
def test_a_voided_plan_defaults_only_while_it_is_owed(voided_after: pd.Timedelta,
                                                      determined: int) -> None:
    row = under_policy(zero_effort_household(), voided_after)
    assert row["approved_orders_user_ever"] == 0
    assert row["never_pay_determined_user"] == determined


def test_never_pay_is_seen_when_victim_reports_label_the_orders_first() -> None:
    """Victim reports on both plans' orders do not settle the account (they are
    reports, not lost disputes), so the never-pay determination must still show."""
    b = zero_effort_household()
    b.report(110, T0 + pd.Timedelta(days=2))
    b.report(111, T0 + pd.Timedelta(days=2))
    row = context_of(b).loc[112]
    assert row["victim_reports_user"] == 2
    assert row["never_pay_determined_user"] == 1


def test_an_order_released_from_a_hold_shares_its_address_from_the_release() -> None:
    """Accounts 1, 2 and 3 each default on one plan at day 44; 2 and 3 ship to one
    address. On day 45 account 1 orders to that address too, and the policy holds the
    order until day 47. Until the release account 1 shares nothing, so its plan is not
    yet never-pay: a decision in between sees what a policy that has not released the
    order yet sees, and the determination comes with the release."""
    b = household()
    b.order(130, 2, T0)
    b.order(131, 3, T0, address=2)
    b.order(132, 1, T0)
    for order_id in (130, 131, 132):
        b.plan(order_id)
    checkout, release = T0 + pd.Timedelta(days=45), T0 + pd.Timedelta(days=47)
    b.order(133, 1, checkout, address=2)
    tables = b.tables()
    at_checkout = asof.PolicyState.approve_all(tables)

    def state(at: pd.Timestamp) -> asof.PolicyState:
        """What the policy has done by ``at``."""
        released = release <= at
        approved = at_checkout.approved
        approved = approved[approved["order_id"] != 133]
        if released:
            approved = pd.concat([approved, pd.DataFrame(
                {"order_id": [133], "approved_at": [release]})], ignore_index=True)
        held = pd.DataFrame({"order_id": [133], "held_at": [checkout],
                             "released_at": [release if released else pd.NaT],
                             "outcome": ["cleared" if released else None],
                             "before_shipment": [True]})
        return dataclasses.replace(
            at_checkout, approved=approved.astype({"approved_at": "datetime64[s]"}),
            held=held.astype({"held_at": "datetime64[s]", "released_at": "datetime64[s]"}))

    waiting, after = checkout + pd.Timedelta(hours=1), release + pd.Timedelta(days=1)
    decisions = pd.DataFrame({"order_id": [133, 133], "decision_at": [waiting, after]})
    final = asof.outcome_columns(tables, state(after), decisions)
    then = asof.outcome_columns(tables, state(waiting), decisions.iloc[:1])
    assert final["never_pay_determined_user"].tolist() == [0, 1]
    pd.testing.assert_frame_equal(final.iloc[:1], then)
