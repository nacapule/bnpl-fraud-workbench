"""Model inputs: which as-of context columns the classifiers read, and labelled rows.

Features are the ``core.asof`` columns tagged for ML; the models compute no history
of their own. Rows are processor-approved order attempts at checkout (the decision
the models score), and a row is labelled only with what was known before a freeze:
an order whose determination is not known by then is left out (unknown is not
negative). The rule here is strict: a label known exactly at the freeze instant
is not yet known.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core import asof
from core.protocol import Window

FEATURES = tuple(column.name for column in asof.COLUMNS if "ml" in column.used_by)

# The depth-3 tree reads four features named before any model was fitted: account
# tenure, how new the device is on the account, how many accounts share the device,
# and the amount against the category's typical amount.
TREE_FEATURES = ("account_age_days", "device_link_age_hours", "accounts_on_device_30d",
                 "amount_over_category_median")


def checkout_rows(context: pd.DataFrame, tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Context rows of processor-approved attempts at their checkout."""
    attempts = tables["order_attempts"]
    approved = attempts.loc[attempts["processor_result"] == "approved", ["order_id", "known_at"]]
    checkout = context.merge(approved.rename(columns={"known_at": "decision_at"}),
                             on=["order_id", "decision_at"], how="inner")
    return checkout.reset_index(drop=True)


def labels_known_before(labels: pd.DataFrame, freeze: pd.Timestamp) -> pd.Series:
    """Each order's label as known strictly before ``freeze`` (orders without one absent)."""
    known = labels[labels["label_known_at"] < pd.Timestamp(freeze)]
    known = known.sort_values(["order_id", "label_known_at"], kind="stable")
    latest = known.drop_duplicates("order_id", keep="last")
    return latest.set_index("order_id")["label"].astype("int64")


def labelled(rows: pd.DataFrame, labels: pd.DataFrame, window: Window,
             known_before: pd.Timestamp) -> pd.DataFrame:
    """Rows decided in ``window`` with their label as known before ``known_before``; rows
    without a known label are dropped."""
    inside = rows[(rows["decision_at"] >= window.start) & (rows["decision_at"] < window.end)]
    known = labels_known_before(labels, known_before)
    out = inside.assign(label=inside["order_id"].map(known))
    return out[out["label"].notna()].astype({"label": "int64"}).reset_index(drop=True)
