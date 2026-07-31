"""Static population: users, devices, cards, addresses, merchants.

Everything is drawn from one seeded numpy Generator passed in by the caller;
no global RNG state, no wall-clock reads — determinism is a tested property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BENIGN_DOMAINS = [
    ("gmail.com", 0.40),
    ("hotmail.com", 0.13),
    ("outlook.com", 0.12),
    ("yahoo.com", 0.10),
    ("icloud.com", 0.09),
    ("proton.me", 0.04),
    ("nortelink.net", 0.06),  # invented ISPs
    ("vistamail.mx", 0.06),
]
DISPOSABLE_DOMAINS = ["mailinator.com", "tempmailo.com", "guerrillamail.com"]

CATEGORIES: list[tuple[str, float, int]] = [
    # (name, lognormal median USD, merchant count weight)
    ("electronics", 420.0, 18),
    ("jewelry", 380.0, 8),
    ("apparel", 85.0, 30),
    ("beauty", 48.0, 16),
    ("home", 140.0, 20),
    ("gaming", 180.0, 10),
    ("sports", 120.0, 12),
    ("shoes", 110.0, 12),
    ("accessories", 60.0, 10),
    ("toys", 55.0, 8),
    ("auto", 160.0, 6),
    ("health", 70.0, 10),
]

UA_FAMILIES = ["iOS", "Android", "Chrome", "Safari", "Firefox"]
COUNTRIES_FOREIGN = ["GB", "DE", "FR", "ES", "BR", "IN", "NG", "RO", "VN", "CN", "RU", "ID"]

FIRST = """olivia liam emma noah amelia oliver sophia elijah isabella lucas mia mason
charlotte ethan harper james evelyn ben luna henry camila alex gianna daniel aria michael
ella jackson sofia sebastian avery david scarlett joseph emily samuel madison john chloe
owen penelope wyatt layla dylan riley luke zoey gabriel nora anthony lily isaac eleanor
grayson hannah jack lillian julian addison levi aubrey christopher ellie andrew stella
joshua natalie theodore zoe caleb leah ryan hazel asher violet nathan aurora thomas""".split()
LAST = """smith johnson williams brown jones garcia miller davis rodriguez martinez
hernandez lopez gonzalez wilson anderson thomas taylor moore jackson martin lee perez
thompson white harris sanchez clark ramirez lewis robinson walker young allen king
wright scott torres nguyen hill flores green adams nelson baker hall rivera campbell
mitchell carter roberts gomez phillips evans turner diaz parker cruz edwards collins""".split()


@dataclass
class Population:
    users: pd.DataFrame
    devices: pd.DataFrame
    user_devices: pd.DataFrame
    cards: pd.DataFrame
    addresses: pd.DataFrame
    merchants: pd.DataFrame
    # convenience lookups used by behavior/patterns
    user_primary_device: dict[int, int] = field(default_factory=dict)
    user_primary_address: dict[int, int] = field(default_factory=dict)
    user_primary_card: dict[int, int] = field(default_factory=dict)


def _emails(rng: np.random.Generator, n: int) -> tuple[list[str], list[str]]:
    domains, weights = zip(*BENIGN_DOMAINS, strict=True)
    w = np.array(weights) / sum(weights)
    dom = rng.choice(domains, n, p=w)
    first = rng.choice(FIRST, n)
    last = rng.choice(LAST, n)
    num = rng.integers(1, 9999, n)
    plus = rng.random(n) < 0.01  # benign plus-addressing exists too
    emails = []
    for i in range(n):
        local = f"{first[i]}.{last[i]}{num[i]}"
        if plus[i]:
            local += "+shop"
        emails.append(f"{local}@{dom[i]}")
    return emails, list(dom)


def build_population(cfg: dict, rng: np.random.Generator) -> Population:
    n_users = cfg["simulator"]["n_users"]
    n_merch = cfg["simulator"]["n_merchants"]
    start = datetime.fromisoformat(cfg["simulator"]["start_date"])
    end = start + timedelta(days=30 * cfg["simulator"]["months"])

    # --- users: signups from 2 years before window start through window end,
    # weighted toward recency so new-account fraud has company
    span_days = (end - start).days + 730
    ages = rng.beta(1.6, 1.0, n_users) * span_days  # skew recent
    signup = [end - timedelta(days=float(a), seconds=float(rng.integers(0, 86400))) for a in ages]
    emails, domains = _emails(rng, n_users)
    users = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "signup_ts": signup,
            "email": emails,
            "email_domain": domains,
            "kyc_country": rng.choice(["US", "CA"], n_users, p=[0.85, 0.15]),
            "dob_year": rng.integers(1955, 2007, n_users),
        }
    )

    # --- devices: everyone has 1, ~25% have 2
    extra = rng.random(n_users) < 0.25
    n_dev = n_users + int(extra.sum())
    devices = pd.DataFrame(
        {
            "device_id": np.arange(1, n_dev + 1),
            "fingerprint": [f"{x:016x}" for x in rng.integers(0, 2**63, n_dev)],
            "ua_family": rng.choice(UA_FAMILIES, n_dev, p=[0.38, 0.34, 0.14, 0.09, 0.05]),
        }
    )
    owner = np.concatenate([np.arange(1, n_users + 1), np.arange(1, n_users + 1)[extra]])
    first_seen = [
        users.signup_ts.iloc[int(u) - 1] + timedelta(days=float(rng.integers(0, 200)))
        for u in owner
    ]
    user_devices = pd.DataFrame(
        {
            "user_id": owner,
            "device_id": devices.device_id.values,
            "first_seen": first_seen,
            "last_seen": [end] * n_dev,
        }
    )

    # --- cards: 1-2 per user; bin_country == kyc mostly
    extra_c = rng.random(n_users) < 0.3
    owner_c = np.concatenate([np.arange(1, n_users + 1), np.arange(1, n_users + 1)[extra_c]])
    n_cards = len(owner_c)
    kyc = users.set_index("user_id").kyc_country
    bin_country = [
        kyc.loc[int(u)] if rng.random() > 0.02 else rng.choice(COUNTRIES_FOREIGN)
        for u in owner_c
    ]
    cards = pd.DataFrame(
        {
            "card_id": np.arange(1, n_cards + 1),
            "user_id": owner_c,
            "bin_country": bin_country,
            "network": rng.choice(["visa", "mc", "amex", "discover"], n_cards,
                                  p=[0.52, 0.32, 0.09, 0.07]),
            "last4": [f"{x:04d}" for x in rng.integers(0, 10000, n_cards)],
        }
    )

    # --- addresses: 1 per user + ~15% second
    extra_a = rng.random(n_users) < 0.15
    owner_a = np.concatenate([np.arange(1, n_users + 1), np.arange(1, n_users + 1)[extra_a]])
    n_addr = len(owner_a)
    cities = ["Minneapolis", "Chicago", "Denver", "Austin", "Phoenix", "Atlanta", "Seattle",
              "Columbus", "Charlotte", "Portland", "Toronto", "Vancouver", "Calgary"]
    addresses = pd.DataFrame(
        {
            "address_id": np.arange(1, n_addr + 1),
            "user_id": owner_a,
            "line_hash": [f"a{x:012x}" for x in rng.integers(0, 2**48, n_addr)],
            "city": rng.choice(cities, n_addr),
            "region": rng.choice(["MN", "IL", "CO", "TX", "AZ", "GA", "WA", "OH", "NC",
                                  "OR", "ON", "BC", "AB"], n_addr),
            "country": [kyc.loc[int(u)] for u in owner_a],
            "added_ts": [users.signup_ts.iloc[int(u) - 1] for u in owner_a],
        }
    )

    # --- merchants
    cat_names = [c[0] for c in CATEGORIES]
    cat_w = np.array([c[2] for c in CATEGORIES], dtype=float)
    cat_w /= cat_w.sum()
    m_cat = rng.choice(cat_names, n_merch, p=cat_w)
    adjectives = ["urban", "nova", "prime", "lux", "peak", "true", "bright", "swift", "pure",
                  "north", "blue", "gold", "iron", "cedar", "atlas"]
    nouns = ["threads", "tech", "goods", "supply", "market", "collective", "haus", "works",
             "trading", "outfitters", "depot", "labs", "gallery", "province", "row"]
    names = []
    used = set()
    while len(names) < n_merch:
        nm = f"{rng.choice(adjectives)}-{rng.choice(nouns)}-{rng.integers(10, 99)}"
        if nm not in used:
            used.add(nm)
            names.append(nm)
    onboard = [start - timedelta(days=float(rng.integers(30, 900))) for _ in range(n_merch)]
    merchants = pd.DataFrame(
        {
            "merchant_id": np.arange(1, n_merch + 1),
            "name": names,
            "category": m_cat,
            "risk_tier": rng.choice([1, 2, 3], n_merch, p=[0.6, 0.3, 0.1]),
            "onboarded_ts": onboard,
        }
    )

    pop = Population(users, devices, user_devices, cards, addresses, merchants)
    pop.user_primary_device = dict(
        user_devices.groupby("user_id").device_id.first()
    )
    pop.user_primary_address = dict(addresses.groupby("user_id").address_id.first())
    pop.user_primary_card = dict(cards.groupby("user_id").card_id.first())
    return pop
