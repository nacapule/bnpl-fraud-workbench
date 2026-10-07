"""The world validator rejects every chronology violation; nothing is clipped or repaired.

Each case starts from the valid mini world and reproduces one way the earlier
generator broke chronology (warm-up orders sampled independently of signup,
devices used before they were first seen, gifts shipped to addresses that did
not exist yet, account events sampled over the whole window, device intervals
that end before they start), plus the other contract rules.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from core import world

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_world"
H = pd.Timedelta


@pytest.fixture(scope="module")
def valid() -> dict[str, pd.DataFrame]:
    return world.read_world(FIXTURE)


def _copy(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: frame.copy() for name, frame in tables.items()}


def _first_order(t, processor_result: str = "approved") -> pd.Series:
    orders = t["order_attempts"]
    return orders[orders["processor_result"] == processor_result].iloc[0]


def order_before_account(t) -> None:
    order = _first_order(t)
    accounts = t["accounts"]
    accounts.loc[accounts["user_id"] == order["user_id"], "created_at"] = (
        order["occurred_at"] + H(days=2)
    )


def order_before_device_link(t) -> None:
    order = _first_order(t)
    links = t["device_links"]
    mask = (links["user_id"] == order["user_id"]) & (links["device_id"] == order["device_id"])
    links.loc[mask, "created_at"] = order["occurred_at"] + H(days=200)


def order_before_address_link(t) -> None:
    order = _first_order(t)
    links = t["address_links"]
    mask = (links["user_id"] == order["user_id"]) & (
        links["address_id"] == order["ship_address_id"]
    )
    links.loc[mask, "created_at"] = order["occurred_at"] + H(days=30)


def order_to_unlinked_address(t) -> None:
    """A gift shipment to another account's address that this account never added."""
    orders = t["order_attempts"]
    other = t["address_links"]
    first = orders.index[0]
    owner = orders.loc[first, "user_id"]
    foreign = other.loc[other["user_id"] != owner, "address_id"].iloc[0]
    orders.loc[first, "ship_address_id"] = foreign


def account_event_before_account(t) -> None:
    events = t["account_events"]
    accounts = t["accounts"].set_index("user_id")
    first = events.index[0]
    signup = accounts.loc[events.loc[first, "user_id"], "created_at"]
    events.loc[first, ["occurred_at", "known_at"]] = signup - H(days=10)


def account_event_before_device_link(t) -> None:
    events = t["account_events"]
    takeover = events[events["kind"] == "password_reset"].index[0]
    events.loc[takeover, ["occurred_at", "known_at"]] = (
        events.loc[takeover, "occurred_at"] - H(hours=1)
    )


def device_link_before_account(t) -> None:
    links = t["device_links"]
    accounts = t["accounts"].set_index("user_id")
    links.loc[0, "created_at"] = accounts.loc[links.loc[0, "user_id"], "created_at"] - H(days=1)


def address_link_before_account(t) -> None:
    links = t["address_links"]
    accounts = t["accounts"].set_index("user_id")
    links.loc[0, "created_at"] = accounts.loc[links.loc[0, "user_id"], "created_at"] - H(days=1)


def link_interval_reversed(t) -> None:
    links = t["device_links"]
    links.loc[0, "removed_at"] = links.loc[0, "created_at"] - H(days=3)


CHRONOLOGY = [
    ("order_before_account", order_before_account),
    ("order_before_device_link", order_before_device_link),
    ("order_before_address_link", order_before_address_link),
    ("order_before_address_link", order_to_unlinked_address),
    ("account_event_before_account", account_event_before_account),
    ("account_event_before_device_link", account_event_before_device_link),
    ("link_before_account", device_link_before_account),
    ("link_before_account", address_link_before_account),
    ("interval_reversed", link_interval_reversed),
]


def known_before_occurred(t) -> None:
    t["payment_attempts"].loc[3, "known_at"] = t["payment_attempts"].loc[3, "occurred_at"] - H(
        seconds=1
    )


def ids_out_of_order(t) -> None:
    attempts = t["payment_attempts"]
    a, b = attempts.index[0], attempts.index[-1]
    attempts.loc[a, "event_id"], attempts.loc[b, "event_id"] = (
        attempts.loc[b, "event_id"], attempts.loc[a, "event_id"]
    )


def order_ids_out_of_order(t) -> None:
    orders = t["order_attempts"]
    orders.loc[0, "occurred_at"], orders.loc[0, "known_at"] = (
        orders.loc[len(orders) - 1, "occurred_at"] + H(seconds=1),
    ) * 2


def duplicate_event_id(t) -> None:
    t["fulfilments"].loc[0, "event_id"] = int(t["deliveries"].loc[0, "event_id"])


def order_after_merchant_closed(t) -> None:
    merchants = t["merchants"]
    closed = merchants["closed_at"].notna()
    merchants.loc[closed, "closed_at"] = merchants.loc[closed, "created_at"] + H(days=1)


def promotion_out_of_validity(t) -> None:
    t["promotions"].loc[0, "valid_to"] = pd.Timestamp("2026-01-01")


def cash_sign(t) -> None:
    cash = t["cash_events"]
    first = cash.index[cash["kind"] == "merchant_settlement"][0]
    cash.loc[first, "amount_cents"] = -cash.loc[first, "amount_cents"]


def cash_before_cause(t) -> None:
    cash = t["cash_events"]
    first = cash.index[cash["kind"] == "recovery"][0]
    cash.loc[first, ["occurred_at"]] = pd.Timestamp("2025-01-01")


def writeoff_before_due(t) -> None:
    writeoffs = t["plan_writeoffs"]
    writeoffs.loc[0, "occurred_at"] = writeoffs.loc[0, "occurred_at"] - H(days=30)


def resolution_before_opening(t) -> None:
    resolutions = t["dispute_resolutions"]
    resolutions.loc[0, "known_at"] = pd.Timestamp("2025-12-01")


def delivery_before_shipment(t) -> None:
    deliveries = t["deliveries"]
    deliveries.loc[0, "occurred_at"] = deliveries.loc[0, "occurred_at"] - H(days=10)


def schedule_does_not_sum(t) -> None:
    t["installment_schedule"].loc[1, "amount_cents"] += 1


def label_before_order(t) -> None:
    t["labels"].loc[0, "label_known_at"] = pd.Timestamp("2024-01-01")


def missing_reference(t) -> None:
    t["order_attempts"].loc[0, "card_id"] = 999_999


def unknown_vocabulary(t) -> None:
    t["dispute_openings"].loc[0, "reason"] = "fraud"


OTHER_RULES = [
    ("known_before_occurred", known_before_occurred),
    ("ids_out_of_order", ids_out_of_order),
    ("ids_out_of_order", order_ids_out_of_order),
    ("duplicate_event_id", duplicate_event_id),
    ("order_outside_merchant", order_after_merchant_closed),
    ("promotion_not_valid", promotion_out_of_validity),
    ("cash_sign", cash_sign),
    ("cash_before_cause", cash_before_cause),
    ("writeoff_before_due", writeoff_before_due),
    ("resolution_before_opening", resolution_before_opening),
    ("delivery_before_fulfilment", delivery_before_shipment),
    ("schedule_mismatch", schedule_does_not_sum),
    ("label_before_order", label_before_order),
    ("missing_reference", missing_reference),
    ("bad_value", unknown_vocabulary),
]


@pytest.mark.parametrize(("check", "mutate"), CHRONOLOGY + OTHER_RULES,
                         ids=[f.__name__ for _, f in CHRONOLOGY + OTHER_RULES])
def test_validator_rejects(valid, check: str, mutate: Callable) -> None:
    tables = _copy(valid)
    mutate(tables)
    before = _copy(tables)
    with pytest.raises(world.WorldError) as raised:
        world.validate_world(tables)
    assert check in raised.value.checks, str(raised.value)
    for name, frame in tables.items():  # nothing clipped or repaired in place
        pd.testing.assert_frame_equal(frame, before[name])


def test_valid_world_passes(valid) -> None:
    assert world.check_world(valid) == []


def test_missing_table_is_reported(valid) -> None:
    tables = _copy(valid)
    del tables["deliveries"]
    with pytest.raises(world.WorldError) as raised:
        world.validate_world(tables)
    assert "missing_table" in raised.value.checks


def test_schema_mismatch_is_reported(valid) -> None:
    tables = _copy(valid)
    tables["orders_extra"] = tables["order_attempts"]
    tables["accounts"] = tables["accounts"].drop(columns="email")
    checks = {v.check for v in world.check_world(tables)}
    assert {"schema", "unknown_table"} <= checks


def test_renumbering_assigns_ids_in_total_order(valid) -> None:
    """Provisional ids with gaps (ties keep their provisional order) map back to 1..N."""
    tables = _copy(valid)
    def remap(ids: pd.Series) -> pd.Series:
        return ids * 7 + 100_000
    provisional = dict(tables)
    for name in world.EVENT_TABLES:
        provisional[name] = tables[name].assign(event_id=remap(tables[name]["event_id"]))
    provisional["payment_reversals"]["payment_event_id"] = remap(
        tables["payment_reversals"]["payment_event_id"]
    )
    provisional["cash_events"]["ref_event_id"] = remap(tables["cash_events"]["ref_event_id"])
    renumbered = world.renumber_events(provisional)
    for name in world.EVENT_TABLES:
        pd.testing.assert_frame_equal(renumbered[name], tables[name], check_dtype=False)
