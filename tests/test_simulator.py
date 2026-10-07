"""The world generator: chronology, determinism, families, construction shortcuts, realism.

Small worlds (a few percent of full scale) are built once per module. The checks
here are written independently of core.world's validator, so the generator is
not judged only by the code that assembled it. The support check on full
development worlds runs with BNPL_FULL_WORLDS=1 (several minutes).
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import config, world
from core.protocol import load_protocol
from simulator import population
from simulator.builder import DAY, Actor, Order
from simulator.fraud import Fraud
from simulator.generate import FAMILIES, build_world, generate_world
from simulator.legit import Customers, Member
from simulator.outcomes import OutcomeParams, Outcomes
from simulator.support import check_support, support_table
from simulator.timing import Clock

SEED = 416
SCALE = 0.04
PROTOCOL = load_protocol()
TEST_START = PROTOCOL.windows["test"].start
Tables = Mapping[str, pd.DataFrame]


@pytest.fixture(scope="module")
def worlds() -> dict[str, dict[str, pd.DataFrame]]:
    return {family: build_world(SEED, family, scale=SCALE)[0] for family in FAMILIES}


@pytest.fixture(scope="module")
def base(worlds) -> dict[str, pd.DataFrame]:
    return worlds["baseline"]


def _orders(tables: Tables) -> pd.DataFrame:
    return tables["order_attempts"].merge(tables["latent_orders"], on="order_id").merge(
        tables["accounts"][["user_id", "created_at", "home_country"]], on="user_id")


# ------------------------------------------------------------ chronology
def chronology_violations(tables: Tables) -> dict[str, int]:
    """Counts of events that use an entity before it exists or outside its link.

    Plain interval checks, independent of core.world: an order needs its account,
    a device linked to the account, a shipping address linked to it and a card on
    it at the order time; an account event needs its account and a linked device;
    a link starts at or after its account and ends after it starts.
    """
    def intervals(frame: pd.DataFrame, item: str) -> dict[tuple[int, int], list[tuple]]:
        out: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        for user, thing, start, end in frame[["user_id", item, "created_at",
                                              "removed_at"]].itertuples(index=False):
            out[(user, thing)].append((start, end))
        return out

    def held(index: dict, user: int, thing: int, at: pd.Timestamp) -> bool:
        return any(start <= at and (pd.isna(end) or at < end)
                   for start, end in index.get((user, thing), ()))

    signup = tables["accounts"].set_index("user_id")["created_at"]
    devices = intervals(tables["device_links"], "device_id")
    addresses = intervals(tables["address_links"], "address_id")
    cards = intervals(tables["cards"], "card_id")
    counts = dict.fromkeys(
        ("order_before_signup", "order_device_not_linked", "order_address_not_linked",
         "order_card_not_on_account", "event_before_signup", "event_device_not_linked",
         "link_before_signup", "link_interval_reversed"), 0)
    for o in tables["order_attempts"].itertuples(index=False):
        at = o.occurred_at
        counts["order_before_signup"] += at < signup[o.user_id]
        counts["order_device_not_linked"] += not held(devices, o.user_id, o.device_id, at)
        counts["order_address_not_linked"] += not held(addresses, o.user_id,
                                                       o.ship_address_id, at)
        counts["order_card_not_on_account"] += not held(cards, o.user_id, o.card_id, at)
    for e in tables["account_events"].itertuples(index=False):
        counts["event_before_signup"] += e.occurred_at < signup[e.user_id]
        counts["event_device_not_linked"] += not held(devices, e.user_id, e.device_id,
                                                      e.occurred_at)
    for name in ("device_links", "address_links", "cards"):
        frame = tables[name]
        counts["link_before_signup"] += int(
            (frame["created_at"] < frame["user_id"].map(signup)).sum())
        counts["link_interval_reversed"] += int((frame["removed_at"] < frame["created_at"]).sum())
    return counts


@pytest.mark.parametrize("family", FAMILIES)
def test_every_event_uses_what_exists_at_its_time(worlds, family) -> None:
    assert chronology_violations(worlds[family]) == dict.fromkeys(
        chronology_violations(worlds[family]), 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_worlds_pass_the_contract_validator(worlds, family) -> None:
    world.validate_world(worlds[family])


def test_events_carry_knowledge_times(base) -> None:
    for name in world.EVENT_TABLES:
        frame = base[name]
        assert (frame["known_at"] >= frame["occurred_at"]).all(), name
        assert (frame["known_at"] <= PROTOCOL.observed_until).all(), name


# ------------------------------------------------------------ determinism
def test_regeneration_is_byte_identical(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    manifest_a = generate_world(SEED, "fraud_mix_shift", first, scale=0.02)
    manifest_b = generate_world(SEED, "fraud_mix_shift", second, scale=0.02)
    assert manifest_a == manifest_b
    files = sorted(p.name for p in first.iterdir())
    assert files == sorted(p.name for p in second.iterdir())
    assert len(files) == len(world.TABLES) + 1
    for name in files:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert manifest_a["code_commit"] is None
    world.verify_manifest(world.read_world(first), manifest_a)


def test_another_seed_gives_another_world() -> None:
    a = build_world(SEED, "baseline", scale=0.02)[0]
    b = build_world(1041, "baseline", scale=0.02)[0]
    assert world.table_sha256("order_attempts", a["order_attempts"]) != world.table_sha256(
        "order_attempts", b["order_attempts"])


# ----------------------------------------------------------- families
def history_before(tables: Tables, at: pd.Timestamp) -> dict[str, bytes]:
    """Everything the platform knew before ``at``, plus the truth of earlier orders."""
    out = {}
    for name, spec in world.TABLES.items():
        frame = tables[name]
        if spec.role == "event":
            frame = frame[frame["known_at"] < at]
        elif spec.created is not None:
            frame = frame[frame[spec.created] < at]
        elif name == "installment_schedule":
            plans = tables["plans"]
            frame = frame[frame["plan_id"].isin(plans.loc[plans["created_at"] < at, "plan_id"])]
        elif name == "labels":
            frame = frame[frame["label_known_at"] < at]
        elif name == "latent_orders":
            orders = tables["order_attempts"]
            frame = frame[frame["order_id"].isin(orders.loc[orders["occurred_at"] < at,
                                                           "order_id"])]
        elif name == "latent_episodes":
            frame = frame[frame["started_at"] < at][["episode_id", "pattern_id", "started_at"]]
            out[name] = frame.to_csv(index=False).encode()
            continue
        elif name == "latent_accounts":
            accounts = tables["accounts"]
            frame = frame[frame["user_id"].isin(accounts.loc[accounts["created_at"] < at,
                                                             "user_id"])]
            out[name] = frame[["user_id", "actor"]].to_csv(index=False).encode()
            continue
        out[name] = world.canonical_csv(name, frame)
    return out


def test_families_share_every_event_before_the_test_window(worlds) -> None:
    baseline = history_before(worlds["baseline"], TEST_START)
    for family in ("acquisition_surge", "fraud_mix_shift"):
        other = history_before(worlds[family], TEST_START)
        assert other.keys() == baseline.keys()
        differ = [name for name in baseline if other[name] != baseline[name]]
        assert differ == [], f"{family} differs before the test window in {differ}"


def test_families_differ_from_the_test_window_on(worlds) -> None:
    def accounts_after(tables: Tables) -> int:
        return int((tables["accounts"]["created_at"] >= TEST_START).sum())

    def episodes_after(tables: Tables, pattern: str) -> int:
        orders = _orders(tables)
        late = orders[(orders["occurred_at"] >= TEST_START) & (orders["pattern_id"] == pattern)]
        return late["episode_id"].nunique()

    base, surge, shift = (worlds[f] for f in FAMILIES)
    assert accounts_after(surge) > 1.5 * accounts_after(base)
    assert episodes_after(shift, "P-ATO") > episodes_after(base, "P-ATO")
    assert episodes_after(shift, "P-STOLEN") > episodes_after(base, "P-STOLEN")
    sleepers = _orders(shift)
    sleepers = sleepers[(sleepers["occurred_at"] >= TEST_START)
                        & (sleepers["pattern_id"] == "P-STOLEN")]
    age = sleepers.groupby("user_id").apply(
        lambda g: (g["occurred_at"].min() - g["created_at"].iloc[0]).days)
    assert (age >= 45).sum() > 0  # aged accounts activated in the test window


# --------------------------------------------- construction shortcuts
def test_home_country_decides_the_ip_country_at_home(base) -> None:
    orders = _orders(base)
    legit = orders[orders["pattern_id"].isna()]
    away = legit["mimic"].fillna("").str.contains("travel")
    at_home = legit[~away]
    assert len(at_home) > 1000
    assert (at_home["ip_country"] == at_home["home_country"]).all()
    assert (legit.loc[away, "ip_country"] != legit.loc[away, "home_country"]).all()
    assert away.sum() > 10
    assert set(orders["home_country"]) == {"US", "CA"}


def test_every_ip_country_is_geolocated(base) -> None:
    countries = set(base["order_attempts"]["ip_country"])
    countries |= set(base["account_events"]["ip_country"])
    assert countries <= set(population.GEO_COUNTRIES)


def test_time_of_day_does_not_reveal_fraud(base) -> None:
    orders = _orders(base)
    night = orders["occurred_at"].dt.hour < 6
    fraud = orders["pattern_id"].notna()
    assert fraud.sum() > 80
    assert abs(night[fraud].mean() - night[~fraud].mean()) < 0.06


def test_ip_ranges_depend_only_on_country(base) -> None:
    events = pd.concat([base["order_attempts"][["ip", "ip_country", "order_id"]],
                        base["account_events"][["ip", "ip_country"]]])
    first = events["ip"].str.split(".").str[0].astype(int)
    blocks = events["ip_country"].map(population.IP_BLOCKS)
    assert all(octet in block for octet, block in zip(first, blocks, strict=True))
    orders = _orders(base)
    orders["octet"] = orders["ip"].str.split(".").str[0].astype(int)
    legit = set(orders.loc[orders["pattern_id"].isna() & (orders["ip_country"] == "US"), "octet"])
    fraud = orders[orders["pattern_id"].notna() & (orders["ip_country"] == "US")]
    assert fraud["octet"].isin(legit).all()


def longest_pattern_run(tables: Tables) -> dict[str, int]:
    """For each pattern, the longest run of consecutive order ids it occupies."""
    labels = tables["latent_orders"].sort_values("order_id")["pattern_id"].fillna("-").to_numpy()
    runs: dict[str, int] = {}
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs[labels[start]] = max(runs.get(labels[start], 0), i - start)
            start = i
    runs.pop("-", None)
    return runs


def test_order_ids_follow_time_not_pattern(worlds) -> None:
    for tables in worlds.values():
        orders = tables["order_attempts"].sort_values("order_id")
        assert orders["occurred_at"].is_monotonic_increasing
        assert max(longest_pattern_run(tables).values()) <= 10


def test_entity_ids_follow_creation_time(base) -> None:
    for name, spec in world.TABLES.items():
        if spec.id_column:
            frame = base[name].sort_values(spec.id_column)
            assert frame[spec.created].is_monotonic_increasing, name


# ------------------------------------------- potential outcomes and labels
def test_the_world_holds_approve_all_outcomes(base) -> None:
    orders = base["order_attempts"]
    approved = orders[orders["processor_result"] == "approved"]
    assert set(base["plans"]["order_id"]) == set(approved["order_id"])
    assert (base["cash_events"]["cause"] == "natural").all()
    assert "refund" not in set(base["cash_events"]["kind"])
    declined = set(orders.loc[orders["processor_result"] == "declined", "order_id"])
    for name in ("fulfilments", "dispute_openings", "victim_reports"):
        assert not set(base[name]["order_id"]) & declined


def test_labels_are_the_policy_determinations(base) -> None:
    from core import config

    recomputed = world.adjudicate(base, horizon_days=PROTOCOL.label_horizon_days,
                                  observed_until=PROTOCOL.observed_until,
                                  **config.load("world")["labels"])
    assert world.canonical_csv("labels", recomputed) == world.canonical_csv(
        "labels", base["labels"])


def test_latent_truth_is_stored_apart(base) -> None:
    latent_columns = {"pattern_id", "episode_id", "intent", "mimic", "actor", "profile"}
    for name in world.OBSERVABLE_TABLES:
        assert not latent_columns & set(base[name].columns), name


# ------------------------------------------------------ realism minimum
def test_legitimate_new_customers_order_at_signup_some_large(base) -> None:
    orders = _orders(base)
    approved = orders[orders["processor_result"] == "approved"].sort_values("occurred_at")
    first = approved.drop_duplicates("user_id")
    legit = first[first["pattern_id"].isna()]
    quick = legit[(legit["occurred_at"] - legit["created_at"]) < pd.Timedelta(hours=1)]
    assert len(quick) > 0.3 * len(legit[legit["created_at"] >= PROTOCOL.order_start])
    median = base["merchants"].set_index("merchant_id")["category"].map(
        {c: m * 100 for c, m in population.CATEGORY_MEDIAN.items()})
    large = quick[quick["amount_cents"] > 2 * quick["merchant_id"].map(median)]
    assert len(large) > 10


def test_households_share_addresses_and_some_devices(base) -> None:
    homes = base["address_links"][base["address_links"]["role"] == "home"]
    sharing = homes.groupby("address_id")["user_id"].nunique()
    assert (sharing >= 2).sum() > 20
    devices = base["device_links"].groupby("device_id")["user_id"].nunique()
    shared = set(devices[devices >= 2].index)
    profiles = base["latent_accounts"].set_index("user_id")["actor"]
    holders = base["device_links"][base["device_links"]["device_id"].isin(shared)]
    assert (holders["user_id"].map(profiles) == "legitimate").sum() > 10


def test_movers_new_phones_resets_and_travel(base) -> None:
    homes = base["address_links"][base["address_links"]["role"] == "home"]
    assert homes["removed_at"].notna().sum() > 20  # moved out
    links = base["device_links"]
    resets = base["account_events"][base["account_events"]["kind"] == "password_reset"]
    paired = resets.merge(links, on=["user_id", "device_id"])
    after_new = paired[(paired["occurred_at"] >= paired["created_at"])
                       & (paired["occurred_at"] - paired["created_at"] < pd.Timedelta(days=1))
                       & (paired["created_at"] > base["accounts"].set_index("user_id")[
                           "created_at"].reindex(paired["user_id"]).to_numpy())]
    actors = base["latent_accounts"].set_index("user_id")["actor"]
    assert (after_new["user_id"].map(actors) == "legitimate").sum() > 5
    mimic = base["latent_orders"]["mimic"].fillna("")
    assert mimic.str.contains("travel").sum() > 10


def test_hardship_includes_new_customers_who_never_pay(base) -> None:
    labels = world.labels_as_of(base["labels"], PROTOCOL.observed_until)
    orders = _orders(base).merge(labels, on="order_id")
    hardship = orders[orders["pattern_id"].isna() & (orders["basis"] == "credit_loss")]
    new = hardship[(hardship["occurred_at"] - hardship["created_at"]) < pd.Timedelta(days=7)]
    assert len(new) > 3
    plans = base["plans"][base["plans"]["order_id"].isin(new["order_id"])]
    paid = base["payment_attempts"]
    paid = paid[(paid["result"] == "success") & (paid["seq"] >= 1)]
    assert (~plans["plan_id"].isin(paid["plan_id"])).any()  # nothing after checkout


def test_fraud_varies_within_patterns(base) -> None:
    orders = _orders(base)
    accounts = base["latent_accounts"]
    profiles = accounts["profile"].fillna("")
    assert profiles.str.contains("sleeper").any()
    for mode in ("single", "burst", "group"):
        assert profiles.str.contains(mode).any(), mode
    ring_accounts = accounts[accounts["actor"] == "synthetic_identity"]
    links = base["device_links"].merge(ring_accounts[["user_id", "episode_id"]], on="user_id")
    devices_per_ring = links.groupby("episode_id")["device_id"].agg(lambda d: d.duplicated().any())
    assert (~devices_per_ring).any() and devices_per_ring.any()  # some rings share no device
    episodes = orders[orders["pattern_id"].notna()].groupby("pattern_id")["episode_id"].nunique()
    assert (episodes >= 2).all()


def test_takeover_victims_are_legitimate_customers(base) -> None:
    orders = _orders(base)
    ato = orders[orders["pattern_id"] == "P-ATO"]
    actors = base["latent_accounts"].set_index("user_id")["actor"]
    assert (ato["user_id"].map(actors) == "legitimate").all()
    earlier = orders[orders["pattern_id"].isna() & (orders["processor_result"] == "approved")]
    first_own = earlier.groupby("user_id")["occurred_at"].min()
    assert (ato["user_id"].map(first_own) < ato["occurred_at"]).all()


def test_fulfilment_lag_follows_the_merchant(base) -> None:
    shipped = base["fulfilments"].merge(base["order_attempts"][["order_id", "occurred_at",
                                                                "merchant_id"]],
                                        on="order_id", suffixes=("", "_order"))
    lag = (shipped["occurred_at"] - shipped["occurred_at_order"]).dt.total_seconds() / 3600
    assert 8 < lag.median() < 16
    per = lag.groupby(shipped["merchant_id"]).median()
    stated = base["merchants"].set_index("merchant_id")["fulfilment_median_hours"]
    busy = shipped["merchant_id"].value_counts()
    busy = busy[busy >= 50].index
    ratio = per[busy] / stated[busy]
    assert ratio.between(0.7, 1.4).all()


def test_merchant_bustouts_close_with_disputes_after(base) -> None:
    closed = base["merchants"][base["merchants"]["closed_at"].notna()]
    assert len(closed) >= 2
    disputes = base["dispute_openings"].merge(
        base["order_attempts"][["order_id", "merchant_id"]], on="order_id").merge(
        closed[["merchant_id", "closed_at"]], on="merchant_id")
    assert (disputes["known_at"] > disputes["closed_at"]).sum() > 0
    labels = base["labels"]
    assert (labels["basis"] == "merchant_bustout").any()


# ------------------------------------------- one behaviour model for every actor
def _clock() -> Clock:
    return Clock(pd.Timestamp("2024-04-01"), pd.Timestamp("2025-09-01"), 1.6, 1.15)


def test_times_are_drawn_inside_their_interval_never_clipped_to_it() -> None:
    clock, rng = _clock(), np.random.default_rng(7)
    lo = clock.start + 5 * DAY + 3_333
    hi = lo + 40 * DAY + 777
    starts = clock.stratified_starts(rng, 160, lo, hi)
    width = (hi - lo) / 160
    assert not {lo + int(k * width) for k in range(161)} & set(starts)
    assert all(lo + int(k * width) <= t < lo + int((k + 1) * width)
               for k, t in enumerate(starts))
    draws = np.array([clock.between(rng, lo, lo + 7_200) for _ in range(5_000)])
    assert draws.min() >= lo and draws.max() < lo + 7_200
    assert (draws == lo).mean() < 0.002
    shop = clock.seasonal_times(rng, 5_000, lo, lo + DAY // 3)
    assert shop.min() >= lo and shop.max() < lo + DAY // 3
    assert (shop == lo).mean() < 0.002 and (shop == lo + DAY // 3 - 1).mean() < 0.002


def test_first_orders_days_after_signup_keep_the_configured_mean() -> None:
    cfg = config.load("world")["customers"]
    customers = Customers.__new__(Customers)
    clock = _clock()
    customers.clock, customers.order_end, customers.c, customers.mu = clock, clock.end, cfg, 0.0
    rng = np.random.default_rng(11)
    signup = clock.start + 30 * DAY
    member = Member(1, signup, signup, "soon", 0.0, False, False, "Ana", "Ruiz")
    delays = [(customers._order_times(member, rng)[0] - signup) / DAY for _ in range(20_000)]
    assert abs(np.mean(delays) - cfg["first_order"]["soon_mean_days"]) < 0.3


def test_takeover_victims_come_from_the_tenured_pool_at_the_stated_share() -> None:
    class Accounts:  # every account usable, as far as the victim choice is concerned
        def approved_before(self, user, t): return True
        def cards_at(self, user, t): return [1]
        def devices_at(self, user, t): return [1]
        def addresses_at(self, user, t, role=None): return [1]

    p = config.load("world")["fraud"]["P-ATO"]
    fraud = Fraud.__new__(Fraud)
    t = 1_000 * DAY
    # half the eligible accounts are tenured, half younger
    fraud._victim_index = sorted(
        [(t - int((p["tenure_days"] + 1 + k % 300) * DAY), k, 2 * t) for k in range(500)]
        + [(t - int((15 + k % (p["tenure_days"] - 20)) * DAY), 500 + k, 2 * t)
           for k in range(500)])
    fraud.b, fraud.f = Accounts(), {"P-ATO": p}
    tenured = []
    for k in range(4_000):
        fraud.victims = set()
        user = fraud._victim(Actor.of(5, 99, k), t)
        tenured.append(user < 500)
    assert abs(np.mean(tenured) - p["tenured_share"]) < 0.03


def test_a_vanishing_merchant_delivers_to_no_buyer() -> None:
    class Shipments:
        merchants = {1: {"median": 12.0, "bustout_from": 900 * DAY, "closed": 910 * DAY}}

        def __init__(self) -> None:
            self.shipped: list[int] = []
            self.delivered: list[int] = []

        def ship(self, a, o, t): self.shipped.append(t)
        def deliver(self, a, o, t): self.delivered.append(t)

    shipments = Shipments()
    out = Outcomes(shipments, _clock(), OutcomeParams.from_config(config.load("world")))
    for k, placed in enumerate((880, 905, 909.5)):
        order = Order.__new__(Order)
        order.merchant, order.t = 1, int(placed * DAY)
        out.fulfil(Actor.of(5, 98, k), order)  # the default path every fraud pattern takes
    assert len(shipments.shipped) == 3 and len(shipments.delivered) == 1
    assert max(shipments.shipped[1:]) < 910 * DAY


def test_collections_on_a_victims_card_succeed_until_the_owner_notices(base) -> None:
    orders = _orders(base)
    third_party = orders.loc[orders["pattern_id"].isin(["P-ATO", "P-STOLEN"]), "order_id"]
    reports = base["victim_reports"].groupby("order_id")["occurred_at"].min()
    unauthorized = base["dispute_openings"]
    unauthorized = unauthorized[unauthorized["reason"] == "unauthorized"]
    notice = pd.concat([reports, unauthorized.groupby("order_id")["occurred_at"].min()])
    notice = notice.groupby(level=0).min()
    notice = notice[notice.index.isin(third_party)]
    pays = base["payment_attempts"].merge(base["plans"][["plan_id", "order_id"]], on="plan_id")
    pays = pays[pays["order_id"].isin(notice.index) & (pays["seq"] > 0)]
    before = pays[pays["occurred_at"] < pays["order_id"].map(notice)]
    assert len(notice) >= 5 and len(before) > 0
    assert (before["result"] == "success").all()


def test_fraud_accounts_use_promotions_and_networks_as_customers_do(base) -> None:
    orders = _orders(base)
    approved = orders[orders["processor_result"] == "approved"].sort_values("occurred_at")
    first = approved.drop_duplicates("user_id")
    first10 = base["promotions"].loc[base["promotions"]["code"] == "FIRST10", "promo_id"]
    opened = ["P-STOLEN", "P-SYNTH", "P-NEVERPAY", "P-INR-ABUSE"]  # accounts fraud opened
    others = first[first["pattern_id"].isin(opened)]
    assert len(others) >= 15 and others["promo_id"].isin(first10).mean() > 0.1
    repeat = approved[approved["pattern_id"].isin(["P-INR-ABUSE", "P-NEVERPAY", "P-SYNTH"])]
    ips = repeat.groupby("user_id")["ip"].nunique()[repeat.groupby("user_id").size() >= 3]
    assert len(ips) >= 3 and (ips > 1).mean() > 0.3


# -------------------------------------------------- support (full worlds)
@pytest.mark.skipif(os.environ.get("BNPL_FULL_WORLDS") != "1",
                    reason="full development worlds take minutes; set BNPL_FULL_WORLDS=1")
@pytest.mark.parametrize("seed", PROTOCOL.development_seeds)
def test_development_worlds_meet_the_support_minimums(seed) -> None:
    tables = build_world(seed, "baseline")[0]
    assert check_support(tables, PROTOCOL) == []
    table = support_table(tables, PROTOCOL)
    fraud = tables["latent_orders"]["pattern_id"].notna().mean()
    assert 0.008 < fraud < 0.015
    assert set(table["pattern"]) == set(world.PATTERNS)


def test_support_table_counts_approved_orders_and_episodes(base) -> None:
    table = support_table(base, PROTOCOL)
    orders = _orders(base)
    stolen = orders[(orders["pattern_id"] == "P-STOLEN")
                    & (orders["processor_result"] == "approved")]
    row = table[(table["pattern"] == "P-STOLEN") & (table["window"] == "all")].iloc[0]
    assert row["approved"] == len(stolen)
    assert row["episodes"] == stolen["episode_id"].nunique()
    assert row["attempts"] > row["approved"] or not np.isnan(row["attempts"])
