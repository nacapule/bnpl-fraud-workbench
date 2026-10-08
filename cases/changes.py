"""One change to the operating policy per case file, run through the replay.

Each change was declared, with its parameters and the mechanism expected, before any of
them ran; each runs once, and its result is published whatever it shows. A change is a
variant of the tuned incumbent built here in code: the same scorer and thresholds with a
rule-score signal that adds, removes or re-routes one condition (and, where the condition
belongs to an evidence family, a reviewer that reads it the same way). Nothing in the
configuration or the frozen code is edited.

:func:`run_changes` replays the unchanged incumbent and every variant on the case world
in the protocol's case cell (base staffing, current layout, policy-specific history,
the evidence reviewer, standard verification, test window) and measures each with the
results' own definitions (``queue_sim.outcomes.outcome_row`` and the recommendation
rule's net contribution, ``core.recommendation.standing``, per 1,000 decided orders).
For each file it records the world's outcomes under both, their difference, the case
order and the account's other orders in the window under both, and, as a latent
diagnostic, the case's episode.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from cases.facts import Run, plain
from cases.rule import RuleError
from core import actions, asof, config, evidence
from core import recommendation as recommendation_module
from core.actions import Disposition
from core.evidence import Condition, Evidence, Family
from model.train import RuleScorer
from queue_sim import outcomes, policies, stage
from queue_sim.replay import PolicyHistory, ReplayResult, Settings
from queue_sim.replay import replay as run_replay
from queue_sim.reviewer import Decision, Reviewer
from queue_sim.roster import ServiceCalendar
from rules import definitions, engine

DAY_HOURS = 24
ILLUSTRATION = ("one replay of one development world, run once with the parameters declared "
                "before it ran: an illustration of the mechanism, not evidence for the change")

# --------------------------------------------------------------------- conditions


def _rules_held(rows: pd.DataFrame) -> pd.DataFrame:
    return evidence.conditions(rows)


def new_device_drop_address(rows: pd.DataFrame) -> np.ndarray:
    """Change 1: an account at least 90 days old, a device first used on it at most 72 h
    before the order, shipping to an address first used at this order that is not home."""
    return ((rows["account_age_days"] >= 90) & (rows["device_link_age_hours"] <= 72)
            & (rows["ship_address_first_use_age_hours"] == 0)
            & (rows["ship_to_home"] == 0)).to_numpy(bool)


def stacked_plan(rows: pd.DataFrame) -> np.ndarray:
    """Change 2: an open plan on an account on which no installment has fallen due."""
    return ((rows["installments_due_user"] == 0)
            & (rows["open_balance_user_cents"] > 0)).to_numpy(bool)


def known_device_home(rows: pd.DataFrame) -> np.ndarray:
    """Change 3: R03 holds, the device was first used on the account at least 90 days
    before, and the order ships to a home address registered at least 90 days before."""
    held = _rules_held(rows)["R03"].to_numpy(bool)
    return held & ((rows["device_link_age_hours"] >= 90 * DAY_HOURS)
                   & (rows["ship_to_home"] == 1)
                   & (rows["home_address_age_days"] >= 90)).to_numpy(bool)


def new_account_device_decline(rows: pd.DataFrame) -> np.ndarray:
    """Change 4: R07 does not hold, but the device had a processor decline in the 24 h
    before and the account is under 7 days old."""
    held = _rules_held(rows)["R07"].to_numpy(bool)
    return ~held & ((rows["processor_declines_device_24h"] >= 1)
                    & (rows["account_age_days"] < 7)).to_numpy(bool)


def linkage_only(rows: pd.DataFrame) -> np.ndarray:
    """Change 5: some §6.2 condition holds and every one that holds is a Linkage condition."""
    held = _rules_held(rows)
    rules = list(evidence.RULE_FAMILY)
    any_held = held[rules].to_numpy(bool).any(axis=1)
    other = [r for r in rules if evidence.RULE_FAMILY[r] is not Family.LINKAGE]
    return any_held & ~held[other].to_numpy(bool).any(axis=1)


# --------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Change:
    """A declared change: what it is, why, what it is expected to do, and how it scores."""

    file: str
    name: str
    declared_for: int  # the case order the motivation describes
    change: str
    motivation: str
    mechanism: str
    parameters: Mapping[str, Any]
    review: Callable[[pd.DataFrame], np.ndarray]  # the review score of the rows
    decline: Callable[[pd.DataFrame], np.ndarray] | None = None  # None: the review score
    evidence: Callable[[Evidence, Mapping[str, Any]], Evidence] | None = None
    noted_after_run: str | None = None  # a correction found after the run, kept apart

    def policy(self, incumbent: policies.Policy) -> policies.Policy:
        base = incumbent.review
        columns = tuple(dict.fromkeys((*base.columns, *evidence.CLASSIFY_COLUMNS,
                                       *self.parameters.get("columns", ()))))
        review = policies.Signal(f"{base.name}+{self.name}", base.version, columns,
                                 self.review)
        decline = review if self.decline is None else policies.Signal(
            f"{base.name}+{self.name}.decline", base.version, columns, self.decline)
        return replace(incumbent, review=review, decline=decline)

    def reviewer(self) -> Reviewer:
        return Reviewer() if self.evidence is None else VariantReviewer(self.evidence)

    def declared(self) -> dict[str, Any]:
        return {"name": self.name, "declared_for_order": self.declared_for,
                "change": self.change, "motivation": self.motivation,
                "mechanism": self.mechanism, "parameters": dict(self.parameters),
                "noted_after_run": self.noted_after_run}


class VariantReviewer(Reviewer):
    """The evidence procedure on evidence a change reads differently (``transform``)."""

    def __init__(self, transform: Callable[[Evidence, Mapping[str, Any]], Evidence]) -> None:
        self.transform = transform

    def decide(self, order_id: int, row: Mapping[str, Any], checks) -> Decision:
        found = self.transform(evidence.classify(row, checks), row)
        allowed = evidence.permitted_actions(found)
        (standard,) = allowed.standard
        disposition = Disposition(standard)
        return Decision(
            disposition=disposition, row=allowed.row, clause=allowed.clauses[standard],
            families=len(found.families_present),
            rules=tuple(c.rule for c in found.conditions),
            checks_to_run=tuple(sorted(allowed.required_checks))
            if disposition is Disposition.HOLD else ())


def _one(row: Mapping[str, Any], test: Callable[[pd.DataFrame], np.ndarray]) -> bool:
    return bool(test(pd.DataFrame([dict(row)]))[0])


def _account_access(found: Evidence, row: Mapping[str, Any]) -> Evidence:
    """Change 1 in the reviewer: the condition is Account access (contact check)."""
    if not _one(row, new_device_drop_address) or any(
            c.family is Family.ACCOUNT_ACCESS for c in found.conditions):
        return found
    condition = Condition("R01", Family.ACCOUNT_ACCESS,
                          ("account_age_days", "device_link_age_hours",
                           "ship_address_first_use_age_hours", "ship_to_home"))
    return replace(found, conditions=(*found.conditions, condition))


def _r03_excepted(found: Evidence, row: Mapping[str, Any]) -> Evidence:
    """Change 3 in the reviewer: R03 does not count under the exception."""
    if not _one(row, known_device_home):
        return found
    return replace(found, conditions=tuple(c for c in found.conditions if c.rule != "R03"))


def _score(rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(engine.score(rows), dtype=float)


W = definitions.WEIGHTS
CHANGES: tuple[Change, ...] = (
    Change(
        file="account_takeover", name="takeover_without_credential_change", declared_for=136685,
        change=("A new Account-access condition, weight 35 (the review band): an account at "
                "least 90 days old, a device first used on it at most 72 h before the order, "
                "and a shipping address first used at this order that is not home. The "
                "reviewer counts it as Account access, so it requires the contact check "
                "(FP-2 §5.2)."),
        motivation=("At decision time the 512-day account had a device first used 0.24 h "
                    "before and shipped to an address first used at this order, not home, "
                    "with no credential change; R01 needs a password or email change, so "
                    "the order reached review only through R03, at the review threshold."),
        mechanism=("Takeovers that leave no credential trail reach review and fail the "
                   "contact check; movers, gift buyers and new phones are held and mostly "
                   "pass, some are cancelled for no response; more review minutes. The "
                   "case's own order scores 70 and is declined at checkout."),
        parameters={"account_age_days_min": 90, "device_first_use_hours_max": 72,
                    "ship_address_first_use_hours": 0, "ship_to_home": 0, "weight": 35,
                    "family": "Account access",
                    "columns": ["account_age_days", "device_link_age_hours",
                                "ship_address_first_use_age_hours", "ship_to_home"]},
        review=lambda rows: _score(rows) + 35.0 * new_device_drop_address(rows),
        evidence=_account_access),
    Change(
        file="never_pay_vs_hardship", name="no_second_plan_before_first_installment",
        declared_for=140378,
        change=("Decline at checkout an order from an account with an open plan on which no "
                "installment has yet fallen due (installments_due_user = 0 and "
                "open_balance_user_cents > 0)."),
        motivation=("At decision time the 2.5-day-old never-pay account placed a second, "
                    "larger order while its first plan was open and no installment had "
                    "fallen due; FP-2 §8.1: repayment cannot be judged before one is due. "
                    "The hardship order was a first order with no open balance."),
        mechanism=("Caps stacked exposure at the first plan, before repayment can be judged; "
                   "declines legitimate new customers who order twice within about two "
                   "weeks. It cannot touch a first order, so it cannot separate a first-order "
                   "hardship default from never-pay."),
        parameters={"installments_due_user": 0, "open_balance_user_cents_above": 0,
                    "action": "auto_decline",
                    "columns": ["installments_due_user", "open_balance_user_cents"]},
        review=lambda rows: _score(rows) + 1000.0 * stacked_plan(rows)),
    Change(
        file="traveller", name="r03_known_device_and_home", declared_for=149795,
        change=("R03 does not count, in the rule score or the family count, when the device "
                "was first used on the account at least 90 days before the order and the "
                "order ships to the account's home address registered at least 90 days "
                "before (modelled on the FP-2 §6.4 household exceptions)."),
        motivation=("At decision time the 224-day account ordered from the device it had "
                    "used since signup (5,370 h) and shipped to its 224-day-old home "
                    "address; only R03 held (card country differs from the IP country, AVS "
                    "failed), which travel explains (FP-2 §6.5(b))."),
        mechanism=("Travellers on their own device shipping home are no longer held: fewer "
                   "legitimate holds and review minutes. Card fraud on the holder's "
                   "long-used device shipping to the holder's home is no longer flagged."),
        parameters={"device_first_use_days_min": 90, "ship_to_home": 1,
                    "home_address_age_days_min": 90, "rule": "R03",
                    "columns": ["device_link_age_hours", "ship_to_home",
                                "home_address_age_days"]},
        review=lambda rows: _score(rows) - W["R03"] * known_device_home(rows),
        evidence=_r03_excepted),
    Change(
        file="card_testing", name="r07_one_device_decline_new_account", declared_for=135235,
        change=("R07 also holds, at its weight of 40 (auto-decline), when the account is under "
                "7 days old and the device had at least 1 processor decline in the 24 h "
                "before the order (otherwise 3)."),
        motivation=("At decision time the account was 35 minutes old with 5 attempts in the "
                    "hour, 3 processor declines on the device, and an order already approved "
                    "within the 24 h; R07 held only at the third device decline."),
        mechanism=("Stops a card-testing session at the first attempt after a decline; "
                   "declines new legitimate customers whose card was declined once. The "
                   "case's own order was already declined and does not change."),
        parameters={"account_age_days_below": 7, "processor_declines_device_24h_min": 1,
                    "weight": 40, "rule": "R07",
                    "columns": ["processor_declines_device_24h", "account_age_days"]},
        review=lambda rows: _score(rows) + W["R07"] * new_account_device_decline(rows),
        noted_after_run=(
            "The row's approved_orders_user_24h = 1 and $448.65 open balance, which the "
            "motivation cites, were approve-all values: the incumbent had auto-declined the "
            "account's 09:33 order (R07 already held, at 3 device declines) 25 minutes "
            "earlier on the same replay day, and the row's outcome-derived columns reflect "
            "the policy's decisions up to the start of the day (the decision's "
            "same_day_orders). No order of this session was approved.")),
    Change(
        file="ring", name="linkage_only_to_review", declared_for=136908,
        change=("An order whose only §6.2 conditions are Linkage conditions goes to review "
                "instead of auto-decline (its decline score is 0; its review score is "
                "unchanged), so an analyst can escalate (FP-2 §5.3(b), §6.6), which blocks "
                "the linked accounts."),
        motivation=("The policy's structure at decision time: the order's only condition was "
                    "R02 (3 accounts on the device in 30 days), and an auto-decline records no "
                    "finding and blocks no account (FP-2 §4.1)."),
        mechanism=("Synthetic identities fail the id_check, are escalated, and their linked "
                   "accounts are blocked before they order again; more review minutes; some "
                   "ring orders ship before review, pass, or go unanswered and are "
                   "cancelled; legitimate households flagged by R02 are held and mostly "
                   "cleared instead of declined."),
        parameters={"families": ["Linkage"], "route": "review"},
        review=_score,
        decline=lambda rows: _score(rows) * ~linkage_only(rows)),
)


# --------------------------------------------------------------------- running


class _Calibrator:
    """The rules model's isotonic calibration, as its metadata records it (unused by
    checkout routing, which compares the raw score with the thresholds)."""

    def __init__(self, thresholds, probabilities) -> None:
        self.thresholds, self.probabilities = list(thresholds), list(probabilities)

    def predict(self, scores) -> np.ndarray:
        return np.interp(np.asarray(scores, float), self.thresholds, self.probabilities)


def incumbent_policy(run: Run) -> policies.Policy:
    """The tuned incumbent as the run used it: the rules model's version from the fit
    stage's metadata and the tuned thresholds; refused unless its version is the run's."""
    meta = json.loads((run.run_dir / "fit" / str(run.rule.seed) / "model_rules.json")
                      .read_text())
    scorer = RuleScorer("rules", meta["version"], tuple(meta["columns"]), engine.score,
                        _Calibrator(meta["calibration"]["thresholds"],
                                    meta["calibration"]["probabilities"]))
    policy = policies.single_scorer(run.rule.replay["policy"], scorer).with_thresholds(
        run.review_threshold, run.decline_threshold)
    if policy.version != run.policy_version:
        raise RuleError(f"the rebuilt incumbent is {policy.version}, the run's is "
                        f"{run.policy_version}")
    return policy


def replay(bench: stage.Bench, policy: policies.Policy, reviewer: Reviewer,
           window: tuple[pd.Timestamp, pd.Timestamp]) -> ReplayResult:
    """``stage.Bench.run`` in the case cell, with the given reviewer."""
    cfg = bench.policy_cfg
    return run_replay(
        bench.world, policy, window=window, roster=stage.base_staffing(cfg).roster(cfg),
        calendar=ServiceCalendar.from_config(cfg), reviewer=reviewer,
        verification=bench.verification("verification"),
        history=PolicyHistory(bench.world, neighbours=bench.neighbours, frozen=bench.frozen),
        settings=Settings.from_config(cfg))


@dataclass
class Measured:
    """One replay and what is reported of it."""

    result: ReplayResult
    row: dict[str, Any]
    rule_net: Any  # Fraction, cents
    cash: pd.DataFrame = field(repr=False)


GROUPS = {
    "cash": ("net_cents", "prevented_loss_cents", "loss_cents", "friction_cost_cents"),
    "adjudicated": ("fraud_orders", "legitimate_orders", "unknown_orders", "legitimate_held",
                    "legitimate_cancelled", "legitimate_declined",
                    "legitimate_declined_checkout", "legitimate_blocked_checkout",
                    "legitimate_declined_review", "fraud_declined_checkout",
                    "fraud_stopped_before_shipping", "fraud_declined_after_shipping"),
    "latent": ("legitimate_truth_orders", "legitimate_truth_held",
               "legitimate_truth_cancelled", "legitimate_truth_declined",
               "legitimate_truth_declined_checkout", "legitimate_truth_blocked_checkout",
               "legitimate_truth_declined_review"),
    "review": ("reviews", "reviews_decided", "review_minutes_offered", "review_minutes_used",
               "available_minutes", "holds", "holds_before_shipping", "escalations",
               "accounts_blocked", "decided_after_shipping"),
}


def _world_block(measured: Measured) -> dict[str, Any]:
    row = measured.row
    out: dict[str, Any] = {"orders": row["orders"], "policy_version": row["policy_version"]}
    for group, columns in GROUPS.items():
        out[group] = {column: row[column] for column in columns}
    out["cash"]["rule_net_cents"] = float(measured.rule_net)
    return out


def _difference(variant: Measured, incumbent: Measured) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group, columns in GROUPS.items():
        out[group] = {column: variant.row[column] - incumbent.row[column] for column in columns}
    if variant.row["orders"] != incumbent.row["orders"]:
        raise RuleError("the variant and the incumbent count different orders")
    out["cash"]["rule_net_cents"] = float(variant.rule_net - incumbent.rule_net)
    out["rule_net_vs_incumbent_per_1000_orders_cents"] = float(
        (variant.rule_net - incumbent.rule_net) * 1000 / variant.row["orders"])
    return out


def _order_view(measured: Measured, order: int) -> dict[str, Any]:
    fate = measured.result.fates.set_index("order_id").loc[order]
    reviews = measured.result.reviews.set_index("order_id")
    routes = measured.result.routes.set_index("order_id")
    review = reviews.loc[order] if order in reviews.index else None
    cash = measured.cash.loc[measured.cash["order_id"] == order, "amount_cents"]
    return {"route": fate["route"], "rule_score": routes.loc[order, "review_score"],
            "first_disposition": None if review is None else review["first_disposition"],
            "final": None if review is None else review["final"],
            "hold_at": fate["hold_at"], "hold_outcome": fate["hold_outcome"],
            "void_at": fate["void_at"], "void_cause": fate["void_cause"],
            "net_cents": int(cash.sum())}


def _orders_block(variant: Measured, incumbent: Measured, orders: list[int],
                  checkout: pd.Series) -> list[dict[str, Any]]:
    out = []
    for order in orders:
        before, after = plain(_order_view(incumbent, order)), plain(_order_view(variant, order))
        changed = any(before[key] != after[key] for key in before if key != "rule_score")
        out.append({"order_id": order, "checkout_at": checkout.loc[order],
                    "incumbent": before, "variant": after, "changed": changed})
    return out


def run_changes(run: Run, facts: Mapping[str, Mapping[str, Any]],
                changes: tuple[Change, ...] = CHANGES,
                log: Callable[[str], None] = lambda text: None,
                check_declared: bool = True) -> dict[str, dict[str, Any]]:
    """Each file's ``tested_change`` block (module docstring), by file name.

    A change's motivation describes the case order it was declared for; with
    ``check_declared`` a selection that gives another order is refused.
    """
    for change in changes:
        entry = facts[change.file] if change.file in facts else None
        selected = None if entry is None else \
            entry["alerts"][entry["primary_alert"]]["publication"]["order_id"]
        if check_declared and selected != change.declared_for:
            raise RuleError(f"{change.name} was declared for order {change.declared_for}; "
                            f"the selection gives {selected}")
    tables = run.tables
    observed = run.protocol.observed_until
    bench = stage.Bench.of(tables, _context(run), seed=run.rule.seed, observed_until=observed)
    window = run.window
    incumbent_policy_ = incumbent_policy(run)
    classes = outcomes.truth(tables, observed)
    legitimate = outcomes.latent_legitimate(tables["latent_orders"])
    rule = recommendation_module.Rule.from_protocol(run.protocol.raw, config.load("policy"))
    keys = {"seed": run.rule.seed, "family": run.rule.family}

    def measure(policy: policies.Policy, reviewer: Reviewer, name: str) -> Measured:
        result = replay(bench, policy, reviewer, window)
        row = outcomes.outcome_row(result, bench.world, keys={**keys, "policy": name,
                                                               "policy_version": policy.version},
                                   ltv_cents=bench.ltv_cents(), classes=classes,
                                   legitimate_truth=legitimate)
        standing = recommendation_module.standing(name, [row], rule)
        cash = actions.policy_cash(tables, result.fates, bench.world.terms,
                                   observed_until=observed)
        return Measured(result, row, standing.net[run.rule.seed],
                        cash.loc[cash["known_at"] <= observed])

    log("replaying the incumbent")
    incumbent = measure(incumbent_policy_, Reviewer(), "incumbent_rules")
    _check_incumbent(run, incumbent)
    fates = incumbent.result.fates.set_index("order_id")
    checkout = fates["checkout_at"]
    latent = tables["latent_orders"].set_index("order_id")
    out = {}
    for change in changes:
        log(f"replaying {change.name}")
        variant = measure(change.policy(incumbent_policy_), change.reviewer(), change.name)
        entry = facts[change.file]
        order = entry["alerts"][entry["primary_alert"]]["publication"]["order_id"]
        others: list[int] = []
        members: list[int] = []
        if order is not None:  # the file's primary slot was filled
            user = int(fates.loc[order, "user_id"])
            others = sorted(int(o) for o in fates.index[(fates["user_id"] == user)
                                                         & (fates.index != order)])
            episode = latent.loc[order, "episode_id"]
            members = [] if pd.isna(episode) else sorted(
                int(o) for o in latent.index[latent["episode_id"] == episode]
                if o in fates.index)
        out[change.file] = plain({
            **change.declared(), "illustration": ILLUSTRATION,
            "result": {
                "policy_version": variant.row["policy_version"],
                "world": {"incumbent": _world_block(incumbent),
                          "variant": _world_block(variant),
                          "difference": _difference(variant, incumbent)},
                "case_order": None if order is None
                else _orders_block(variant, incumbent, [int(order)], checkout)[0],
                "account_orders": _orders_block(variant, incumbent, others, checkout),
                "latent_episode": _episode(variant, incumbent, members),
            },
            "superseded": [],
        })
    return out


def _episode(variant: Measured, incumbent: Measured, members: list[int]) -> dict[str, Any] | None:
    """The case's episode in the window under both (simulation truth, a diagnostic)."""
    if not members:
        return None

    def summary(measured: Measured) -> dict[str, Any]:
        fates = measured.result.fates.set_index("order_id").loc[members]
        cash = measured.cash.loc[measured.cash["order_id"].isin(members), "amount_cents"]
        routes = fates["route"].value_counts().sort_index()
        return {"routes": {str(k): int(v) for k, v in routes.items()},
                "voided": int(fates["void_at"].notna().sum()),
                "net_cents": int(cash.sum())}

    return {"orders_in_window": len(members), "incumbent": summary(incumbent),
            "variant": summary(variant)}


def _context(run: Run) -> pd.DataFrame:
    return asof.build_context(run.tables)


def _check_incumbent(run: Run, incumbent: Measured) -> None:
    """The incumbent replayed here is the run's: the same fates as it kept, and the same
    outcome row as its replay stage published when the case world was evaluated."""
    kept = run.fates.reset_index(drop=True)
    mine = incumbent.result.fates.reset_index(drop=True)
    if not kept.equals(mine):
        raise RuleError("the incumbent replayed for the tested changes differs from the run's")
    path = run.results_dir / "replay.json"
    if not path.exists():
        return
    rows = [row for row in json.loads(path.read_text())["tables"]["replay.outcomes"]
            if row.get("seed") == run.rule.seed and row.get("family") == run.rule.family
            and row.get("policy") == run.rule.replay["policy"]
            and row.get("capacity_level") == run.rule.replay["capacity_level"]
            and row.get("layout") == run.rule.replay["layout"]
            and (row.get("history"), row.get("reviewer"), row.get("verification"))
            == stage.VARIANTS[0]]
    for row in rows:
        different = sorted(column for column in outcomes.OUTCOME_COLUMNS
                           if row.get(column) != incumbent.row[column])
        if different:
            raise RuleError(f"the incumbent's outcome row differs from the run's in {different}")
