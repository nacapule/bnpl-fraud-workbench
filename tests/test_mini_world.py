"""The mini world: a readable fixture in the world contract (tests/fixtures/mini_world)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from core import ledger, world

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_world"


def _builder():
    spec = importlib.util.spec_from_file_location("mini_world_build", FIXTURE / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return world.read_world(FIXTURE)


def _order(tables: dict[str, pd.DataFrame], email: str, amount_cents: int) -> int:
    accounts = tables["accounts"].set_index("email")["user_id"]
    orders = tables["order_attempts"]
    mine = orders["user_id"] == accounts[email]
    match = orders[mine & (orders["amount_cents"] == amount_cents)]
    assert len(match) == 1, (email, amount_cents)
    return int(match["order_id"].iloc[0])


def test_committed_files_equal_a_fresh_build() -> None:
    builder = _builder()
    built = builder.build()
    for name in world.TABLES:
        committed = (FIXTURE / f"{name}.csv").read_bytes()
        assert world.canonical_csv(name, built[name]) == committed, name
    committed = json.loads((FIXTURE / "manifest.json").read_text())
    assert builder.manifest(built) == committed


def test_mini_world_is_valid_and_matches_its_manifest(tables) -> None:
    world.validate_world(tables)
    world.verify_manifest(tables, json.loads((FIXTURE / "manifest.json").read_text()))


def test_every_table_and_event_kind_is_present(tables) -> None:
    for name in world.TABLES:
        assert len(tables[name]) > 0, name
    assert set(tables["account_events"]["kind"]) == set(
        world.TABLES["account_events"].column("kind").values
    )
    natural_kinds = set(ledger.CASH_KINDS) - {"refund"}  # refunds come only from actions
    assert set(tables["cash_events"]["kind"]) == natural_kinds
    assert set(tables["order_attempts"]["processor_result"]) == {"approved", "declined"}
    assert set(tables["dispute_resolutions"]["outcome"]) == {"won", "lost"}


@pytest.mark.parametrize(
    ("email", "amount", "net"),
    [
        # ana, $84.50 repaid: fee 422.5 rounds to 423, settlement -8027, payments +8450
        ("ana.88@gmail.com", 8450, 423),
        # ana, $430 repaid; installment 1 bounced (-10750) and was paid again (+10750):
        # -40850 + 43000 - 10750 + 10750
        ("ana.88@gmail.com", 43000, 2150),
        # ben, $1,299.99 with 10% promotion (13000): settlement -(129999 - 6500 - 13000),
        # promotion -13000, principal 116999 repaid: -110499 - 13000 + 116999
        ("ben.99@gmail.com", 129999, -6500),
        # eve, stolen card: -61750 settlement, +16250 down payment, -16250 and -1500 for the
        # lost unauthorized dispute, write-off of 48750 recovers 9750
        ("eve.95@gmail.com", 65000, -53500),
        # fay, never pays after checkout: -49400 + 13000 + recovery 7800 (20% of 39000)
        ("fay.1@gmail.com", 52000, -28600),
        # hal, claim rejected: -7125 + 7500, debit -3750 then credit +3750, fee -1500
        ("hal.90@gmail.com", 7500, -1125),
        # ivy, parcel lost, claim upheld, merchant open: -19950 + 21000 - 10500 - 1500 + 10500
        ("ivy.84@gmail.com", 21000, -450),
        # jay, merchant closed before the claim was upheld, no recourse:
        # -74100 + 78000 - 39000 - 1500
        ("jay.86@gmail.com", 78000, -36600),
        # dan, taken over: -85405 + 22475 + recovery 13485 (20% of 67425)
        ("dan.71@gmail.com", 89900, -49445),
    ],
)
def test_order_cash_matches_hand_calculation(tables, email: str, amount: int, net: int) -> None:
    cash = tables["cash_events"]
    order_id = _order(tables, email, amount)
    assert ledger.net_cents(cash[cash["order_id"] == order_id]) == net


def test_declined_attempts_have_no_plan_cash_or_label(tables) -> None:
    orders = tables["order_attempts"]
    declined = set(orders.loc[orders["processor_result"] == "declined", "order_id"])
    assert len(declined) == 2
    for name in ("plans", "cash_events", "labels", "fulfilments"):
        assert not declined & set(tables[name]["order_id"]), name


@pytest.mark.parametrize(
    ("email", "amount", "expected"),
    [
        ("dan.71@gmail.com", 89900, [(1, "account_takeover", "2025-02-09 10:00:00")]),
        # stolen card: no never-pay marker, so the lost unauthorized dispute decides
        ("eve.95@gmail.com", 65000, [(1, "third_party_fraud", "2025-04-10 09:00:00")]),
        # two zero-effort plans three days apart: known at the later default
        # (installment 1 of the second plan due 02-18 10:00, unpaid 30 days later)
        ("fay.1@gmail.com", 52000, [(1, "never_pay", "2025-03-20 10:00:00")]),
        ("fay.1@gmail.com", 18000, [(1, "never_pay", "2025-03-20 10:00:00")]),
        # three accounts on one phone default within days: known at the third default
        ("nr1.98@gmail.com", 32000, [(1, "never_pay", "2025-04-21 16:20:00")]),
        ("nr3.98@gmail.com", 41000, [(1, "never_pay", "2025-04-21 16:20:00")]),
        # an unmarked zero-effort default is a credit loss
        ("gus.80@gmail.com", 24000, [(0, "credit_loss", "2025-03-26 21:00:00")]),
        # clean at the horizon, then the second rejected claim
        ("hal.90@gmail.com", 7500, [(0, "no_finding", "2025-02-08 18:00:00"),
                                    (1, "inr_abuse", "2025-03-10 10:00:00")]),
        ("hal.90@gmail.com", 9500, [(1, "inr_abuse", "2025-03-10 10:00:00")]),
        # dispute pending at the horizon postpones the negative to its resolution
        ("ivy.84@gmail.com", 21000, [(0, "no_finding", "2025-03-20 09:00:00")]),
        # claim upheld after the merchant stopped trading
        ("jay.86@gmail.com", 78000, [(1, "merchant_bustout", "2025-04-05 10:00:00")]),
        ("kim.93@gmail.com", 41000, [(0, "no_finding", "2025-03-11 11:00:00")]),
        # household of two on one promotion is not promotion abuse
        ("ben.99@gmail.com", 129999, [(0, "no_finding", "2025-03-13 18:40:00")]),
        # three accounts on one device take the promotion and never come back:
        # known 90 days after the third use
        ("pf1.0@gmail.com", 5500, [(0, "no_finding", "2025-05-09 20:20:00"),
                                   (1, "promo_abuse", "2025-06-10 21:20:00")]),
        ("pf3.0@gmail.com", 6000, [(0, "no_finding", "2025-05-11 21:20:00"),
                                   (1, "promo_abuse", "2025-06-10 21:20:00")]),
        # too recent: no label by the end of observation
        ("mo.97@outlook.com", 3000, []),
    ],
)
def test_adjudicated_labels(tables, email: str, amount: int, expected: list) -> None:
    labels = tables["labels"]
    rows = labels[labels["order_id"] == _order(tables, email, amount)]
    got = [(int(r.label), r.basis, str(r.label_known_at)) for r in rows.itertuples()]
    assert got == expected


def _relabel(tables: dict[str, pd.DataFrame], observed_until: str | None = None) -> pd.DataFrame:
    builder = _builder()
    return world.adjudicate(tables, horizon_days=builder.HORIZON_DAYS,
                            observed_until=observed_until or builder.OBSERVED_UNTIL,
                            **builder.LABEL_RULES)


def _bases(labels: pd.DataFrame, order_id: int) -> list[str]:
    return labels.loc[labels["order_id"] == order_id, "basis"].tolist()


def test_never_pay_needs_two_other_sharing_accounts(tables) -> None:
    changed = dict(tables)
    links = tables["device_links"]
    nr3 = tables["accounts"].set_index("email").loc["nr3.98@gmail.com", "user_id"]
    changed["device_links"] = links[links["user_id"] != nr3]
    labels = _relabel(changed)
    assert _bases(labels, _order(tables, "nr1.98@gmail.com", 32000)) == ["credit_loss"]


def test_never_pay_ring_through_one_email_identity(tables) -> None:
    """Variants of one mailbox mark the ring just as a shared phone does."""
    changed = dict(tables)
    accounts = tables["accounts"].copy()
    ring = accounts["email"].str.startswith(("nr2.", "nr3."))
    accounts.loc[ring, "email"] = ["N.R1.98+b@gmail.com", "nr1.98+c@googlemail.com"]
    changed["accounts"] = accounts
    links = tables["device_links"]
    ring_device = links.loc[links["user_id"].isin(accounts.loc[ring, "user_id"]), "device_id"]
    changed["device_links"] = links[~links["device_id"].isin(ring_device)]
    labels = _relabel(changed)
    assert _bases(labels, _order(tables, "nr1.98@gmail.com", 32000)) == ["never_pay"]


def _with_plain_reorder(tables, email: str, amount: int, at: str) -> dict:
    """The account orders again without a promotion at ``at``."""
    changed = dict(tables)
    orders = tables["order_attempts"]
    plain = orders[orders["order_id"] == _order(tables, email, amount)].copy()
    plain["order_id"] = orders["order_id"].max() + 1
    plain["event_id"] = 10**6
    plain["occurred_at"] = plain["known_at"] = pd.Timestamp(at)
    plain["promo_id"] = pd.NA
    plain["promo_discount_cents"] = 0
    changed["order_attempts"] = pd.concat([orders, plain], ignore_index=True)
    return changed


def test_promo_abuse_needs_no_plain_reorder_within_90_days(tables) -> None:
    labels = _relabel(_with_plain_reorder(tables, "pf2.0@gmail.com", 5800, "2025-04-20 12:00"))
    assert "promo_abuse" not in labels["basis"].tolist()


def test_promo_reorder_window_is_each_accounts_own(tables) -> None:
    """pf1 reorders 96 days after its own use: outside its window, so all three still count."""
    labels = _relabel(_with_plain_reorder(tables, "pf1.0@gmail.com", 5500, "2025-06-14 20:20"))
    for email, amount in (("pf1.0@gmail.com", 5500), ("pf3.0@gmail.com", 6000)):
        assert _bases(labels, _order(tables, email, amount)) == ["no_finding", "promo_abuse"]


def test_promo_uses_more_than_90_days_apart_do_not_group(tables) -> None:
    changed = dict(tables)
    orders = tables["order_attempts"].copy()
    late = orders["order_id"] == _order(tables, "pf3.0@gmail.com", 6000)
    orders.loc[late, ["occurred_at", "known_at"]] = pd.Timestamp("2025-06-20 21:20")
    changed["order_attempts"] = orders
    assert "promo_abuse" in _relabel(tables, "2026-12-31")["basis"].tolist()
    assert "promo_abuse" not in _relabel(changed, "2026-12-31")["basis"].tolist()


def test_inr_abuse_needs_carrier_confirmed_delivery(tables) -> None:
    changed = dict(tables)
    deliveries = tables["deliveries"]
    hal1 = _order(tables, "hal.90@gmail.com", 7500)
    changed["deliveries"] = deliveries[deliveries["order_id"] != hal1]
    assert "inr_abuse" not in _relabel(changed)["basis"].tolist()


def test_inr_abuse_waits_for_the_last_delivery_confirmation(tables) -> None:
    changed = dict(tables)
    deliveries = tables["deliveries"].copy()
    hal2 = _order(tables, "hal.90@gmail.com", 9500)
    late = pd.Timestamp("2025-03-15 08:00")
    deliveries.loc[deliveries["order_id"] == hal2, ["occurred_at", "known_at"]] = late
    changed["deliveries"] = deliveries
    labels = _relabel(changed)
    rows = labels[(labels["basis"] == "inr_abuse")]
    assert sorted(rows["order_id"]) == sorted([_order(tables, "hal.90@gmail.com", 7500), hal2])
    assert set(rows["label_known_at"]) == {late}


def test_bustout_needs_no_carrier_confirmed_delivery(tables) -> None:
    changed = dict(tables)
    deliveries = tables["deliveries"]
    jay1 = _order(tables, "jay.86@gmail.com", 78000)
    delivered = deliveries.iloc[[0]].assign(order_id=jay1, event_id=10**6)
    delivered["occurred_at"] = delivered["known_at"] = pd.Timestamp("2025-02-14 12:00")
    changed["deliveries"] = pd.concat([deliveries, delivered], ignore_index=True)
    assert "merchant_bustout" not in _relabel(changed)["basis"].tolist()


def test_promo_abuse_ignores_shared_addresses(tables) -> None:
    """Households share addresses; only a shared device or email links promotion users."""
    changed = dict(tables)
    links = tables["device_links"]
    farm = tables["devices"].set_index("fingerprint").loc["fp-d_farm", "device_id"]
    farm_links = links[links["device_id"] == farm]
    shared_address = tables["address_links"].iloc[[0]]
    moved = [shared_address.assign(user_id=user) for user in farm_links["user_id"]]
    changed["device_links"] = links[links["device_id"] != farm]
    changed["address_links"] = pd.concat([tables["address_links"], *moved], ignore_index=True)
    assert "promo_abuse" not in _relabel(changed)["basis"].tolist()


def test_bustout_needs_the_merchant_closed_before_the_claim_is_upheld(tables) -> None:
    changed = dict(tables)
    merchants = tables["merchants"].copy()
    merchants.loc[merchants["closed_at"].notna(), "closed_at"] = pd.Timestamp("2025-04-30")
    changed["merchants"] = merchants
    labels = _relabel(changed)
    assert "merchant_bustout" not in labels["basis"].tolist()


def test_labels_are_reproduced_from_observable_tables_only(tables) -> None:
    observable = {name: tables[name] for name in world.OBSERVABLE_TABLES}
    builder = _builder()
    again = world.adjudicate(observable, horizon_days=builder.HORIZON_DAYS,
                             observed_until=builder.OBSERVED_UNTIL, **builder.LABEL_RULES)
    pd.testing.assert_frame_equal(again, tables["labels"])


def test_unknown_is_not_negative(tables) -> None:
    order = _order(tables, "hal.90@gmail.com", 7500)
    def label_at(at: str):
        known = world.labels_as_of(tables["labels"], at)
        row = known[known["order_id"] == order]
        return None if row.empty else int(row["label"].iloc[0])
    assert label_at("2025-02-01 00:00:00") is None
    assert label_at("2025-02-09 00:00:00") == 0
    assert label_at("2025-03-11 00:00:00") == 1


def test_event_ids_follow_the_total_order(tables) -> None:
    events = world.event_frame(tables)
    assert events["event_id"].tolist() == list(range(1, len(events) + 1))
