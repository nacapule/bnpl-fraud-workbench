"""Point-in-time feature construction from the simulator CSV frames.

All temporal lookups are evaluated at each order timestamp.  In particular,
installment outcomes are never used: a payment only counts once ``paid_ts`` is
at or before the order being scored.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "account_age_days",
    "n_orders_1h",
    "n_orders_24h",
    "n_orders_7d",
    "amount_sum_24h",
    "amount_sum_7d",
    "n_orders_24h_device",
    "distinct_devices_30d",
    "distinct_addresses_ever",
    "amount_over_category_median",
    "avs_mismatch",
    "cvv_mismatch",
    "bin_ip_country_mismatch",
    "n_prior_avs_mismatches",
    "prior_installments_due",
    "prior_installments_paid_share",
    "hours_since_last_credential_change",
    "n_accounts_on_device_30d",
    "n_accounts_on_ship_address",
    "email_domain_class",
    "promo_redemptions_before",
    "is_first_order",
    "night_order",
    "ship_addr_is_new",
]

COMMON_EMAIL_DOMAINS = {
    "gmail.com",
    "hotmail.com",
    "icloud.com",
    "outlook.com",
    "proton.me",
    "yahoo.com",
}
DISPOSABLE_EMAIL_DOMAINS = {
    "guerrillamail.com",
    "mailinator.com",
    "tempmailo.com",
}

_READ_SPECS: dict[str, tuple[list[str], list[str]]] = {
    "orders": (
        [
            "order_id",
            "user_id",
            "merchant_id",
            "ts",
            "amount",
            "ip_country",
            "device_id",
            "card_id",
            "ship_address_id",
            "avs_result",
            "cvv_result",
            "status",
        ],
        ["ts"],
    ),
    "users": (["user_id", "signup_ts", "email_domain"], ["signup_ts"]),
    "cards": (["card_id", "bin_country"], []),
    "addresses": (["address_id", "added_ts"], ["added_ts"]),
    "merchants": (["merchant_id", "category"], []),
    "plans": (["plan_id", "order_id"], []),
    "installments": (["plan_id", "due_ts", "paid_ts"], ["due_ts", "paid_ts"]),
    "account_events": (["user_id", "ts", "kind"], ["ts"]),
    "promo_redemptions": (["user_id", "ts"], ["ts"]),
    "labels": (["order_id", "pattern_id"], []),
}


def load_feature_frames(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load only columns used by :func:`build_features`."""
    root = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for name, (columns, dates) in _READ_SPECS.items():
        frames[name] = pd.read_csv(
            root / f"{name}.csv",
            usecols=columns,
            parse_dates=dates or None,
        )
    return frames


def _datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if not isinstance(values.dtype, pd.DatetimeTZDtype) and values.dtype.kind != "M":
        values = pd.to_datetime(values, errors="coerce")
    # Pandas 3 commonly parses CSV datetimes at microsecond resolution.  Force
    # nanoseconds so the integer window constants below have one stable unit.
    return values.astype("datetime64[ns]")


def _group_slices(values: np.ndarray):
    """Yield contiguous slices; callers sort by the group column first."""
    if not len(values):
        return
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    for start, stop in zip(
        np.r_[0, edges],
        np.r_[edges, len(values)],
        strict=True,
    ):
        yield slice(int(start), int(stop))


def _asof_event_count(
    target_user: np.ndarray,
    target_ns: np.ndarray,
    event_users: np.ndarray,
    event_ns: np.ndarray,
    *,
    include_current: bool,
) -> np.ndarray:
    """Count events for each user's targets with a vectorized search per user."""
    result = np.zeros(len(target_user), dtype=np.int64)
    if not len(event_users):
        return result

    event_order = np.lexsort((event_ns, event_users))
    event_users = event_users[event_order]
    event_ns = event_ns[event_order]
    target_order = np.lexsort((target_ns, target_user))
    sorted_target_users = target_user[target_order]
    sorted_target_ns = target_ns[target_order]

    event_groups = {
        int(event_users[s.start]): event_ns[s]
        for s in _group_slices(event_users)
    }
    side = "right" if include_current else "left"
    sorted_result = np.zeros(len(target_user), dtype=np.int64)
    for target_slice in _group_slices(sorted_target_users):
        user_id = int(sorted_target_users[target_slice.start])
        history = event_groups.get(user_id)
        if history is not None:
            sorted_result[target_slice] = np.searchsorted(
                history,
                sorted_target_ns[target_slice],
                side=side,
            )
    result[target_order] = sorted_result
    return result


def _hours_since_event(
    target_user: np.ndarray,
    target_ns: np.ndarray,
    event_users: np.ndarray,
    event_ns: np.ndarray,
) -> np.ndarray:
    """Hours since the latest event, capped at 10,000 when old or absent."""
    result = np.full(len(target_user), 10_000.0)
    if not len(event_users):
        return result

    event_order = np.lexsort((event_ns, event_users))
    event_users = event_users[event_order]
    event_ns = event_ns[event_order]
    target_order = np.lexsort((target_ns, target_user))
    sorted_users = target_user[target_order]
    sorted_ns = target_ns[target_order]
    event_groups = {
        int(event_users[s.start]): event_ns[s]
        for s in _group_slices(event_users)
    }
    sorted_result = np.full(len(target_user), 10_000.0)
    for target_slice in _group_slices(sorted_users):
        history = event_groups.get(int(sorted_users[target_slice.start]))
        if history is None:
            continue
        positions = np.searchsorted(history, sorted_ns[target_slice], side="right") - 1
        present = positions >= 0
        if present.any():
            target_positions = np.arange(target_slice.start, target_slice.stop)[present]
            elapsed = (
                sorted_ns[target_positions] - history[positions[present]]
            ) / 3_600_000_000_000
            sorted_result[target_positions] = np.minimum(elapsed, 10_000.0)
    result[target_order] = sorted_result
    return result


def _add_order_history(features: pd.DataFrame) -> pd.DataFrame:
    """Add order histories in O(n log n), plus linear distinct counters."""
    ordered = features.sort_values(["user_id", "ts", "order_id"], kind="stable").reset_index(
        drop=True
    )
    n = len(ordered)
    user = ordered["user_id"].to_numpy()
    ts_ns = ordered["ts"].astype("int64").to_numpy()
    amount = ordered["amount"].to_numpy(dtype=float)
    avs_bad = ordered["avs_mismatch"].to_numpy(dtype=np.int64)
    device = ordered["device_id"].to_numpy()
    address = ordered["ship_address_id"].to_numpy()

    windows = {
        "n_orders_1h": 3_600_000_000_000,
        "n_orders_24h": 24 * 3_600_000_000_000,
        "n_orders_7d": 7 * 24 * 3_600_000_000_000,
    }
    counts = {name: np.zeros(n, dtype=np.int32) for name in windows}
    amount_24h = np.zeros(n, dtype=float)
    amount_7d = np.zeros(n, dtype=float)
    distinct_devices = np.zeros(n, dtype=np.int32)
    distinct_addresses = np.zeros(n, dtype=np.int32)
    prior_avs = np.zeros(n, dtype=np.int32)
    first_order = np.zeros(n, dtype=np.int8)

    for group_slice in _group_slices(user):
        times = ts_ns[group_slice]
        values = amount[group_slice]
        positions = np.arange(group_slice.start, group_slice.stop)
        for name, width in windows.items():
            left = np.searchsorted(times, times - width, side="left")
            counts[name][group_slice] = np.arange(len(times)) - left + 1

        cumulative = np.cumsum(values)
        for width, output in (
            (24 * 3_600_000_000_000, amount_24h),
            (7 * 24 * 3_600_000_000_000, amount_7d),
        ):
            left = np.searchsorted(times, times - width, side="left")
            sums = cumulative.copy()
            has_prior = left > 0
            sums[has_prior] -= cumulative[left[has_prior] - 1]
            output[group_slice] = sums

        prior_avs[group_slice] = np.cumsum(avs_bad[group_slice]) - avs_bad[group_slice]
        first_order[group_slice.start] = 1

        device_counts: Counter[int] = Counter()
        left = 0
        for local_position, absolute_position in enumerate(positions):
            while times[left] < times[local_position] - 30 * 24 * 3_600_000_000_000:
                old_device = int(device[group_slice][left])
                device_counts[old_device] -= 1
                if device_counts[old_device] == 0:
                    del device_counts[old_device]
                left += 1
            device_counts[int(device[absolute_position])] += 1
            distinct_devices[absolute_position] = len(device_counts)

        seen_addresses: set[int] = set()
        for absolute_position in positions:
            seen_addresses.add(int(address[absolute_position]))
            distinct_addresses[absolute_position] = len(seen_addresses)

    for name, values in counts.items():
        ordered[name] = values
    ordered["amount_sum_24h"] = amount_24h
    ordered["amount_sum_7d"] = amount_7d
    ordered["distinct_devices_30d"] = distinct_devices
    ordered["distinct_addresses_ever"] = distinct_addresses
    ordered["n_prior_avs_mismatches"] = prior_avs
    ordered["is_first_order"] = first_order
    return ordered


def _rolling_distinct_accounts(
    frame: pd.DataFrame,
    group_column: str,
    *,
    window_ns: int | None,
) -> np.ndarray:
    """Distinct accounts seen on a device/address as of every order."""
    sorted_frame = frame.sort_values(
        [group_column, "ts", "order_id"], kind="stable"
    ).reset_index()
    result = np.zeros(len(frame), dtype=np.int32)
    groups = sorted_frame[group_column].to_numpy()
    users = sorted_frame["user_id"].to_numpy()
    times = sorted_frame["ts"].astype("int64").to_numpy()

    for group_slice in _group_slices(groups):
        counts: Counter[int] = Counter()
        left = group_slice.start
        for position in range(group_slice.start, group_slice.stop):
            if window_ns is not None:
                while times[left] < times[position] - window_ns:
                    old_user = int(users[left])
                    counts[old_user] -= 1
                    if counts[old_user] == 0:
                        del counts[old_user]
                    left += 1
            counts[int(users[position])] += 1
            original_position = int(sorted_frame.at[position, "index"])
            result[original_position] = len(counts)
    return result


def _installment_history(
    orders: pd.DataFrame,
    plans: pd.DataFrame,
    installments: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    target_user = orders["user_id"].to_numpy(dtype=np.int64)
    target_ns = orders["ts"].astype("int64").to_numpy()
    if plans.empty or installments.empty:
        return np.zeros(len(orders), dtype=np.int64), np.zeros(len(orders), dtype=float)

    plan_users = plans[["plan_id", "order_id"]].merge(
        orders[["order_id", "user_id"]], on="order_id", how="inner", validate="many_to_one"
    )
    history = installments[["plan_id", "due_ts", "paid_ts"]].merge(
        plan_users[["plan_id", "user_id"]], on="plan_id", how="inner", validate="many_to_one"
    )
    history["due_ts"] = _datetime(history, "due_ts")
    history["paid_ts"] = _datetime(history, "paid_ts")
    history = history[history["due_ts"].notna()]
    event_users = history["user_id"].to_numpy(dtype=np.int64)
    due_ns = history["due_ts"].astype("int64").to_numpy()
    due = _asof_event_count(
        target_user,
        target_ns,
        event_users,
        due_ns,
        include_current=True,
    )

    paid = history[history["paid_ts"].notna()].copy()
    # Both facts must be known: an installment cannot count as paid history
    # before it is due, even if an input happens to contain an early payment.
    paid_effective = paid[["due_ts", "paid_ts"]].max(axis=1).astype("int64").to_numpy()
    paid_count = _asof_event_count(
        target_user,
        target_ns,
        paid["user_id"].to_numpy(dtype=np.int64),
        paid_effective,
        include_current=True,
    )
    share = np.divide(
        paid_count,
        due,
        out=np.zeros(len(orders), dtype=float),
        where=due > 0,
    )
    return due, share


def build_features(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    devices: pd.DataFrame | None = None,
    user_devices: pd.DataFrame | None = None,
    cards: pd.DataFrame | None = None,
    addresses: pd.DataFrame | None = None,
    merchants: pd.DataFrame | None = None,
    plans: pd.DataFrame | None = None,
    installments: pd.DataFrame | None = None,
    payments: pd.DataFrame | None = None,
    account_events: pd.DataFrame | None = None,
    promos: pd.DataFrame | None = None,
    promo_redemptions: pd.DataFrame | None = None,
    chargebacks: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    *,
    as_of: str = "per-order",
) -> pd.DataFrame:
    """Build one deterministic, point-in-time feature row per approved order.

    Unused frames are accepted so callers can pass the complete CSV world as
    keyword arguments.  Static dimensions are safe to join, while every
    timestamped fact is filtered by an as-of search.
    """
    del devices, user_devices, payments, promos, chargebacks
    if as_of != "per-order":
        raise ValueError("build_features only supports as_of='per-order'")
    required = {
        "cards": cards,
        "addresses": addresses,
        "merchants": merchants,
        "plans": plans,
        "installments": installments,
        "account_events": account_events,
        "promo_redemptions": promo_redemptions,
        "labels": labels,
    }
    missing = [name for name, frame in required.items() if frame is None]
    if missing:
        raise ValueError(f"missing required frames: {', '.join(missing)}")
    assert cards is not None
    assert addresses is not None
    assert merchants is not None
    assert plans is not None
    assert installments is not None
    assert account_events is not None
    assert promo_redemptions is not None
    assert labels is not None

    base = orders.copy()
    base["ts"] = _datetime(base, "ts")
    if base["ts"].isna().any():
        raise ValueError("orders.ts contains invalid timestamps")
    base = base.sort_values(["ts", "order_id"], kind="stable").reset_index(drop=True)
    base["amount"] = pd.to_numeric(base["amount"], errors="raise").astype(float)
    base["avs_mismatch"] = base["avs_result"].eq("N").astype(np.int8)
    base["cvv_mismatch"] = base["cvv_result"].eq("N").astype(np.int8)

    users_static = users[["user_id", "signup_ts", "email_domain"]].copy()
    users_static["signup_ts"] = _datetime(users_static, "signup_ts")
    address_static = addresses[["address_id", "added_ts"]].copy().rename(
        columns={"address_id": "ship_address_id", "added_ts": "ship_address_added_ts"}
    )
    address_static["ship_address_added_ts"] = _datetime(
        address_static, "ship_address_added_ts"
    )
    base = base.merge(users_static, on="user_id", how="left", validate="many_to_one")
    base = base.merge(
        cards[["card_id", "bin_country"]], on="card_id", how="left", validate="many_to_one"
    )
    base = base.merge(
        address_static, on="ship_address_id", how="left", validate="many_to_one"
    )
    base = base.merge(
        merchants[["merchant_id", "category"]],
        on="merchant_id",
        how="left",
        validate="many_to_one",
    )
    if base[["bin_country", "category"]].isna().any().any():
        raise ValueError("static dimensions do not cover all orders")

    user_known = base["signup_ts"].notna() & base["signup_ts"].le(base["ts"])
    base["account_age_days"] = (
        (base["ts"] - base["signup_ts"]).dt.total_seconds() / 86_400
    ).where(user_known, 0.0).clip(lower=0)
    base["bin_ip_country_mismatch"] = base["bin_country"].ne(base["ip_country"]).astype(np.int8)
    base["night_order"] = base["ts"].dt.hour.between(0, 5).astype(np.int8)
    address_age_hours = (
        base["ts"] - base["ship_address_added_ts"]
    ).dt.total_seconds() / 3_600
    base["ship_addr_is_new"] = address_age_hours.between(0, 48, inclusive="left").astype(
        np.int8
    )
    domain = base["email_domain"].str.lower().where(user_known)
    base["email_domain_class"] = np.select(
        [domain.isin(DISPOSABLE_EMAIL_DOMAINS), domain.isin(COMMON_EMAIL_DOMAINS)],
        [2, 0],
        default=1,
    ).astype(np.int8)

    base = _add_order_history(base)
    base["n_orders_24h_device"] = 0
    device_sorted = base.sort_values(["device_id", "ts", "order_id"], kind="stable")
    device_counts = np.zeros(len(base), dtype=np.int32)
    device_values = device_sorted["device_id"].to_numpy()
    device_times = device_sorted["ts"].astype("int64").to_numpy()
    for group_slice in _group_slices(device_values):
        times = device_times[group_slice]
        left = np.searchsorted(times, times - 24 * 3_600_000_000_000, side="left")
        positions = np.arange(len(times)) - left + 1
        device_counts[device_sorted.index.to_numpy()[group_slice]] = positions
    base["n_orders_24h_device"] = device_counts

    base["n_accounts_on_device_30d"] = _rolling_distinct_accounts(
        base,
        "device_id",
        window_ns=30 * 24 * 3_600_000_000_000,
    )
    base["n_accounts_on_ship_address"] = _rolling_distinct_accounts(
        base,
        "ship_address_id",
        window_ns=None,
    )

    category_sorted = base.sort_values(["category", "ts", "order_id"], kind="stable").copy()
    category_sorted["_category_median"] = category_sorted.groupby(
        "category", sort=False
    )["amount"].transform(lambda values: values.expanding().median())
    category_median = category_sorted["_category_median"].reindex(base.index)
    base["amount_over_category_median"] = base["amount"] / category_median.clip(lower=0.01)

    due, paid_share = _installment_history(base, plans, installments)
    base["prior_installments_due"] = due
    base["prior_installments_paid_share"] = paid_share

    events = account_events.copy()
    events["ts"] = _datetime(events, "ts")
    credentials = events[
        events["kind"].isin(["password_change", "email_change"]) & events["ts"].notna()
    ]
    base["hours_since_last_credential_change"] = _hours_since_event(
        base["user_id"].to_numpy(dtype=np.int64),
        base["ts"].astype("int64").to_numpy(),
        credentials["user_id"].to_numpy(dtype=np.int64),
        credentials["ts"].astype("int64").to_numpy(),
    )

    redemption = promo_redemptions.copy()
    redemption["ts"] = _datetime(redemption, "ts")
    redemption = redemption[redemption["ts"].notna()]
    base["promo_redemptions_before"] = _asof_event_count(
        base["user_id"].to_numpy(dtype=np.int64),
        base["ts"].astype("int64").to_numpy(),
        redemption["user_id"].to_numpy(dtype=np.int64),
        redemption["ts"].astype("int64").to_numpy(),
        include_current=False,
    )

    labeled_orders = pd.Index(labels["order_id"].drop_duplicates())
    base["label"] = base["order_id"].isin(labeled_orders).astype(np.int8)
    approved = base[base["status"].str.lower().eq("approved")].copy()
    output_columns = ["order_id", "user_id", "ts", "amount", "label", *FEATURE_COLUMNS]
    return approved[output_columns].sort_values(["ts", "order_id"], kind="stable").reset_index(
        drop=True
    )
