"""Generate one synthetic world: python -m simulator.generate --seed 416 --family baseline

The world covers the protocol's order horizon (experiments/protocol.yaml) and
holds every attempted order with the outcomes it would have if every order were
approved; only processor declines happen inside it. It is written as one CSV
per table of the world contract (core/world.py) plus ``manifest.json``.

Randomness comes from independent streams, one per component and actor
(``numpy.random.SeedSequence(seed, spawn_key=(component, index))``). A family
adds actors from its own streams from its start date onwards, so every world of
a seed shares its history before the test window whatever the family.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import config, ledger, world
from core.protocol import Protocol, load_protocol
from simulator import population as pop
from simulator.builder import DAY, Actor, Builder
from simulator.fraud import MERCH, Fraud
from simulator.legit import Customers, Household
from simulator.merchants import Market, legit_merchants
from simulator.outcomes import OutcomeParams, Outcomes
from simulator.timing import Clock, seconds

REPO = Path(__file__).resolve().parent.parent
FAMILIES = ("baseline", "acquisition_surge", "fraud_mix_shift", "lag_half", "lag_double")
LAG_FAMILIES = ("lag_half", "lag_double")  # fulfilment-lag sensitivity from the test window

# Random-stream components (SeedSequence spawn keys); never renumber them.
MERCHANTS, ARRIVALS, HOUSEHOLDS, PROMOTIONS = 1, 2, 3, 4
SURGE_ARRIVALS, SURGE_HOUSEHOLDS = 20, 21


def build_world(seed: int, family: str, *, scale: float = 1.0,
                cfg: dict[str, Any] | None = None,
                protocol: Protocol | None = None) -> tuple[dict[str, pd.DataFrame], list[int]]:
    """The world's tables (in the core.world contract) and its bust-out merchant ids."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if scale <= 0:
        raise ValueError("scale must be positive")
    cfg = config.load("world") if cfg is None else cfg
    protocol = load_protocol() if protocol is None else protocol
    terms = ledger.ProductTerms.from_config(cfg)
    order_start, order_end = seconds(protocol.order_start), seconds(protocol.order_end)
    test_start = seconds(protocol.windows["test"].start)
    clock = Clock(protocol.order_start, protocol.order_end,
                  cfg["calendar"]["holiday_lift"], cfg["calendar"]["weekend_lift"])
    target = cfg["volume"]["target_orders"] * scale

    b = Builder(terms)
    market = Market(b)
    m = cfg["merchants"]
    legit_merchants(b, market, Actor.of(seed, MERCHANTS, 0),
                    max(20, int(round(cfg["volume"]["merchants"] * min(1.0, 4 * scale)))),
                    order_start, order_end, seconds(m["trading_since"]), m["onboarding_share"],
                    cfg)
    promotions = _promotions(b, Actor.of(seed, PROMOTIONS, 0), cfg["promotions"])
    outcomes = Outcomes(b, clock, OutcomeParams.from_config(cfg))
    if family in LAG_FAMILIES:
        outcomes.scale_lag(test_start, cfg["families"][family]["fulfilment_lag_factor"])
    customers = Customers(b, clock, market, outcomes, cfg, promotions, order_end)
    fraud = Fraud(b, clock, market, outcomes, customers, cfg, seed, promotions, scale)

    # merchants that bust out, and the customers they acquire
    bustouts = [fraud.merchant_bustout(a, closed, 1_000_000 * (k + 1))
                for k, (a, closed) in enumerate(fraud._schedule(
                    MERCH, fraud.count("P-MERCH"), order_start + 80 * DAY, order_end))]

    # the customer base: households open before the horizon, then new arrivals
    c = cfg["customers"]
    n_households = int(round(c["households_per_1000_orders"] * target / 1000))
    n_existing = int(round(n_households * c["existing_household_share"]))
    arrivals = Actor.of(seed, ARRIVALS, 0)
    since = seconds(c["existing_since"])
    existing = np.sort(since + (np.sqrt(arrivals.rng.random(n_existing))
                                * (order_start - since)).astype(np.int64))
    new = clock.seasonal_times(arrivals.rng, n_households - n_existing, order_start, order_end)
    sizes = tuple((int(k), float(v)) for k, v in c["household_size"].items())
    for index, created in enumerate([*existing.tolist(), *new.tolist()]):
        h = _household(seed, HOUSEHOLDS, index + 1, int(created))
        customers.skeleton(h, int(pop.pick(h.actor.rng, sizes)), existing=created < order_start)
        customers.base.append(h)
    families = cfg["families"]
    fraud.sleeper_reserve(_scaled(families["fraud_mix_shift"]["sleeper_accounts"], scale),
                          test_start - 300 * DAY, test_start - 45 * DAY)
    customers.calibrate(target * (1 - cfg["volume"]["target_fraud_share"]))
    customers.finish_registry()
    for h in customers.base:
        customers.live(h)
    for h in customers.extra:
        customers.live(h)
    fraud.run()

    if family == "acquisition_surge":
        spec = families["acquisition_surge"]
        expected = len(new) * clock.exposure(test_start, order_end) / clock.exposure(
            order_start, order_end)
        surge = Actor.of(seed, SURGE_ARRIVALS, 0)
        n = int(surge.rng.poisson((spec["new_customer_multiplier"] - 1) * expected))
        households = []
        for index, created in enumerate(clock.seasonal_times(surge.rng, n, test_start, order_end)):
            h = _household(seed, SURGE_HOUSEHOLDS, index + 1, int(created))
            customers.skeleton(h, int(pop.pick(h.actor.rng, sizes)), existing=False)
            for member in h.members:
                member.promo_take = spec["first_purchase_take"]
                member.tags.add("campaign")
            households.append(h)
        customers.finish_registry()
        for h in households:
            customers.live(h)
    elif family == "fraud_mix_shift":
        spec = families["fraud_mix_shift"]
        share = (order_end - test_start) / (order_end - order_start - 30 * DAY)
        extra = int(round((spec["takeover_multiplier"] - 1) * fraud.count("P-ATO") * share))
        fraud.extra_takeovers(extra, test_start, order_end - DAY)
        fraud.activate_sleepers(test_start, order_end - 3 * DAY)

    _unique_ties(b)
    tables = b.tables(seconds(protocol.observed_until))
    tables["cash_events"] = ledger.derive_cash_events(tables, terms)
    tables = world.renumber_events(tables)
    tables["cash_events"] = world.coerce("cash_events", tables["cash_events"])
    tables["labels"] = world.adjudicate(
        tables, horizon_days=protocol.label_horizon_days,
        observed_until=protocol.observed_until, **cfg["labels"])
    ordered = {name: world.coerce(name, tables[name]) for name in world.TABLES}
    return ordered, sorted(b.ids[pk] for pk in bustouts)


def generate_world(seed: int, family: str, out_dir: Path, *, scale: float = 1.0,
                   validate: bool = True) -> dict[str, Any]:
    """Generate, validate and write a world; return its manifest.

    Writes one ``<table>.csv`` per table (core.world.write_world) and
    ``manifest.json``. ``code_commit`` is left empty: the pipeline records the
    commit in its lineage. ``scale`` shrinks the world (tests and CI).
    """
    cfg = config.load("world")
    protocol = load_protocol()
    tables, bustouts = build_world(seed, family, scale=scale, cfg=cfg, protocol=protocol)
    if validate:
        world.validate_world(tables)
    out = Path(out_dir)
    world.write_world(tables, out)
    manifest = world.build_manifest(
        tables,
        generator_version=cfg["generator_version"],
        config={"world": cfg, "scale": scale},
        seed=seed,
        family=family,
        order_start=str(protocol.order_start),
        order_end=str(protocol.order_end),
        observed_until=str(protocol.observed_until),
        bustout_merchant_ids=bustouts,
    )
    world.write_manifest(manifest, out / "manifest.json")
    return manifest


# ---------------------------------------------------------------- helpers
def _household(seed: int, component: int, index: int, created: int) -> Household:
    actor = Actor.of(seed, component, index)
    return Household(index, actor, created, pop.pick(actor.rng, pop.HOME_COUNTRIES))


def _promotions(b: Builder, a: Actor, specs: list[dict[str, Any]]) -> dict[str, int]:
    return {spec["code"]: b.promotion(a, spec["code"], int(spec["discount_bps"]),
                                      bool(spec["first_purchase_only"]), seconds(spec["from"]),
                                      seconds(spec["to"]))
            for spec in specs}


def _scaled(count: int, scale: float) -> int:
    return max(2, int(round(count * scale)))


def _unique_ties(b: Builder) -> None:
    """Provisional event ids are random 62-bit draws; refuse the (vanishingly rare)
    world in which two collide rather than break a tie arbitrarily."""
    seen: set[int] = set()
    lists = [b.account_events, b.payment_attempts, b.reversals, b.fulfilments, b.deliveries,
             b.openings, b.resolutions, b.victim_reports, b.writeoffs]
    for o in b.orders:
        if o.tie in seen:
            raise RuntimeError("duplicate order tiebreak")  # not reachable in practice
        seen.add(o.tie)
    for rows in lists:
        for row in rows:
            if row[0] in seen:
                raise RuntimeError("duplicate event tiebreak")
            seen.add(row[0])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="default: the canonical seed")
    parser.add_argument("--family", default="baseline", choices=FAMILIES)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: data/worlds/<seed>-<family>[-scale<scale>]")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)
    protocol = load_protocol()
    seed = protocol.canonical_seed if args.seed is None else args.seed
    if seed in protocol.final_seeds:
        from core.protocol import check_seed

        check_seed(seed, REPO)
    suffix = "" if args.scale == 1.0 else f"-scale{args.scale:g}"
    out = args.out or REPO / "data" / "worlds" / f"{seed}-{args.family}{suffix}"
    started = time.perf_counter()
    manifest = generate_world(seed, args.family, out, scale=args.scale,
                              validate=not args.no_validate)
    elapsed = time.perf_counter() - started
    print(f"world {seed}/{args.family} (scale {args.scale:g}) written to {out} "
          f"in {elapsed:.1f}s")
    for name, entry in manifest["tables"].items():
        print(f"  {name:22s} {entry['rows']:>9,d}")
    from simulator.support import support_report

    print(support_report(world.read_world(out, ["order_attempts", "latent_orders",
                                                "latent_episodes", "labels"]), protocol))


if __name__ == "__main__":
    sys.exit(main())
