"""Reference tables shared by every actor: categories, names, emails, countries, IPs.

Nothing here depends on whether an actor is legitimate or fraudulent. Emails,
IP addresses and devices are drawn by the same functions for everyone; an IP
address depends only on its country (F25), and a customer's home country
decides the country of their home IP (F13).
"""

from __future__ import annotations

import numpy as np

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
    # (name, median price in dollars, merchant count weight)
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
CATEGORY_MEDIAN = {name: median for name, median, _ in CATEGORIES}
RESALE_CATEGORIES = ("electronics", "jewelry", "gaming")  # easy to resell

MOBILE_UA = ("iOS", "Android")
DESKTOP_UA = ("Chrome", "Safari", "Firefox", "Edge")
CARD_NETWORKS = (("visa", 0.52), ("mc", 0.32), ("amex", 0.09), ("discover", 0.07))

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

# Countries the platform geolocates. Every IP country the world emits has a
# centroid in rules/engine.py (R11) and db/queries/Q10.
HOME_COUNTRIES = (("US", 0.85), ("CA", 0.15))
CITIES = {
    "US": [("Minneapolis", "MN"), ("Chicago", "IL"), ("Denver", "CO"), ("Austin", "TX"),
           ("Houston", "TX"), ("Phoenix", "AZ"), ("Atlanta", "GA"), ("Seattle", "WA"),
           ("Columbus", "OH"), ("Charlotte", "NC"), ("Portland", "OR"), ("Miami", "FL"),
           ("Nashville", "TN"), ("Tulsa", "OK"), ("Boston", "MA"), ("San Diego", "CA")],
    "CA": [("Toronto", "ON"), ("Vancouver", "BC"), ("Calgary", "AB"), ("Montreal", "QC"),
           ("Ottawa", "ON"), ("Winnipeg", "MB")],
}
# Where legitimate customers travel, and where foreign fraud traffic comes from.
TRAVEL_COUNTRIES = (("GB", 0.22), ("FR", 0.16), ("ES", 0.16), ("DE", 0.12), ("IN", 0.10),
                    ("BR", 0.08), ("CN", 0.06), ("VN", 0.05), ("ID", 0.05))
FOREIGN_FRAUD_COUNTRIES = (("RO", 0.18), ("NG", 0.16), ("VN", 0.12), ("RU", 0.12),
                           ("BR", 0.10), ("IN", 0.10), ("CN", 0.08), ("ID", 0.06),
                           ("GB", 0.05), ("DE", 0.03))
STOLEN_CARD_ISSUERS = (("US", 0.62), ("CA", 0.10), ("GB", 0.10), ("DE", 0.05), ("FR", 0.05),
                       ("BR", 0.04), ("ES", 0.04))

# IPv4 first octets per country. Every actor draws from the same blocks, so the
# address range reveals the country and nothing else.
IP_BLOCKS = {
    "US": (24, 47, 50, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 96, 97, 98, 99,
           104, 107, 108, 162, 166, 172, 173, 174, 184, 199, 204, 208),
    "CA": (24, 64, 70, 99, 142, 174, 184, 199, 205, 206, 207),
    "GB": (2, 25, 31, 51, 62, 77, 81, 86, 90, 92, 109, 151, 176, 212),
    "DE": (5, 31, 37, 46, 78, 79, 80, 84, 87, 91, 93, 95, 178, 217),
    "FR": (2, 5, 37, 46, 62, 77, 78, 80, 81, 82, 86, 88, 90, 109, 176, 212),
    "ES": (2, 5, 31, 37, 46, 62, 77, 79, 80, 81, 83, 85, 88, 95, 176, 213),
    "BR": (138, 143, 152, 167, 177, 179, 186, 187, 189, 191, 200, 201),
    "IN": (1, 14, 27, 42, 49, 59, 103, 106, 115, 117, 122, 157, 182, 203),
    "NG": (41, 102, 105, 129, 154, 160, 196, 197),
    "RO": (5, 31, 37, 46, 79, 82, 86, 89, 92, 93, 109, 176, 188, 194),
    "VN": (1, 14, 27, 42, 58, 103, 113, 115, 116, 117, 118, 123, 171, 183),
    "CN": (1, 14, 27, 36, 39, 42, 58, 59, 60, 61, 101, 106, 110, 111, 112, 113, 114, 115),
    "RU": (5, 31, 37, 46, 77, 78, 79, 80, 81, 83, 85, 87, 89, 91, 92, 93, 94, 95, 176, 178),
    "ID": (27, 36, 39, 49, 103, 110, 112, 114, 116, 118, 120, 125, 139, 180, 182, 202),
}
GEO_COUNTRIES = tuple(IP_BLOCKS)


def pick(rng: np.random.Generator, table: tuple[tuple[str, float], ...] | list) -> str:
    """One value from a ``((value, weight), ...)`` table."""
    u = rng.random() * sum(weight for _, weight in table)
    for value, weight in table:
        u -= weight
        if u < 0:
            return value
    return table[-1][0]


def travel_destination(rng: np.random.Generator, home: str, neighbour_share: float) -> str:
    """Where a customer living in ``home`` travels: often the neighbouring country."""
    if rng.random() < neighbour_share:
        return {"US": "CA", "CA": "US"}.get(home, "US")
    return pick(rng, TRAVEL_COUNTRIES)


def ip_address(rng: np.random.Generator, country: str) -> str:
    """An address from the country's blocks; the same function serves every actor."""
    blocks = IP_BLOCKS[country]
    first = blocks[int(rng.integers(0, len(blocks)))]
    return (f"{first}.{int(rng.integers(0, 256))}.{int(rng.integers(0, 256))}."
            f"{int(rng.integers(1, 255))}")


def person_name(rng: np.random.Generator) -> tuple[str, str]:
    return FIRST[int(rng.integers(0, len(FIRST)))], LAST[int(rng.integers(0, len(LAST)))]


def email_address(
    rng: np.random.Generator,
    first: str,
    last: str,
    *,
    disposable: bool = False,
    domain: str | None = None,
) -> str:
    """An email in one of the common local-part styles people use."""
    if domain is None:
        domain = (DISPOSABLE_DOMAINS[int(rng.integers(0, len(DISPOSABLE_DOMAINS)))]
                  if disposable else pick(rng, BENIGN_DOMAINS))
    number = int(rng.integers(1, 9999))
    style = int(rng.integers(0, 5))
    local = (f"{first}.{last}{number}", f"{first}{last}{number % 100}",
             f"{first[0]}.{last}{number}", f"{first}{number}",
             f"{last}.{first}{number % 1000}")[style]
    if rng.random() < 0.01:
        local += "+shop"
    return f"{local}@{domain}"


def email_variant(rng: np.random.Generator, root_email: str, n: int) -> str:
    """Another spelling of ``root_email`` that normalizes to the same identity:
    a plus-tag, or for Gmail an inserted dot."""
    local, domain = root_email.split("@")
    local = local.split("+")[0]
    if domain == "gmail.com" and rng.random() < 0.5:
        bare = local.replace(".", "")
        cut = 1 + int(rng.integers(0, max(1, len(bare) - 1)))
        return f"{bare[:cut]}.{bare[cut:]}@{domain}" if n % 2 else f"{bare}+{n}@{domain}"
    return f"{local}+{n}@{domain}"


def device_ua(rng: np.random.Generator, mobile: bool = True) -> str:
    table = MOBILE_UA if mobile else DESKTOP_UA
    return table[int(rng.integers(0, len(table)))]


def fingerprint(rng: np.random.Generator) -> str:
    return f"{int(rng.integers(0, 2**62)):016x}"


def card_network(rng: np.random.Generator) -> str:
    return pick(rng, CARD_NETWORKS)


def last4(rng: np.random.Generator) -> str:
    return f"{int(rng.integers(0, 10000)):04d}"


def home_city(rng: np.random.Generator, country: str) -> tuple[str, str]:
    cities = CITIES[country]
    return cities[int(rng.integers(0, len(cities)))]


def line_hash(rng: np.random.Generator) -> str:
    return f"h{int(rng.integers(0, 2**60)):015x}"


def order_amount_cents(
    rng: np.random.Generator, category: str, multiplier: float = 1.0, sigma: float = 0.65
) -> int:
    """A price drawn around the category median (lognormal), in cents."""
    dollars = float(np.exp(rng.normal(np.log(CATEGORY_MEDIAN[category] * multiplier), sigma)))
    return int(round(min(max(dollars, 12.0), 4000.0) * 100))
