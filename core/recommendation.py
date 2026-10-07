"""The pre-registered recommendation rule, applied to the replay's outcome rows.

The rule is ``reporting.recommendation_rule`` in ``experiments/protocol.yaml``;
:meth:`Rule.from_protocol` reads its numbers from there, and :func:`apply_rule`
applies it in one operating cell at a time: a world family, a replay cell (capacity
level, shift layout, history, reviewer, verification) and the LTV proxy the friction
cost uses. In each cell:

1. **Eligibility**, from each policy's rows over the seeds it was evaluated on. Lost
   legitimate customers (declined at checkout, blocked, declined after review, or
   cancelled after an unanswered hold) and legitimate orders held (asked to verify),
   per 10,000 legitimate orders: at most a cap on the mean over seeds and on every
   seed. At each priority P0-P3, the mean over seeds of the share of queue entries
   decided within the target: at least a floor, assessed only when the entries pooled
   over seeds reach a minimum (fewer, or no queue at all: reported, not assessed).
   When the incumbent fails a criterion, that criterion becomes "no worse than the
   incumbent" for the challengers, and the cell records the incumbent's misses.
2. **The hurdle**: a challenger's rule net contribution minus the incumbent's, per
   1,000 attempted orders and paired by seed, must have a mean of at least the hurdle
   and be strictly positive on enough of the paired seeds (``positive_seeds``, for
   example 9 of 10; with fewer seeds than the rule lists, every seed).
3. **The choice**: among eligible challengers that clear the hurdle, the highest mean
   improvement; exact ties go to fewer lost customers, then fewer review minutes, then
   the simpler policy. Otherwise the incumbent stays, and the best challenger (the
   best eligible one, else the best of all) is named with its improvement.

The rule's net contribution is the ledger's net cash after the friction cost (the LTV
proxy on each legitimate order declined or cancelled) less the opportunity cost of the
analyst allotment: its minutes in the window, used or not, at the loaded hourly cost.
A policy with no review route releases the allotment and is charged nothing. A cell
with another LTV proxy recomputes the friction cost from the same rows.

The arithmetic is exact (fractions), so a tie is a tie and a bound is met or not. No
p-value is computed anywhere here: the selected policy is chosen among the
challengers, and the sign bar is a consistency rule across simulated worlds, not a
significance test.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from core.results import Metric, SeedSpread

INCUMBENT = "incumbent_rules"
PRIORITIES = ("p0", "p1", "p2", "p3")
CRITERIA = ("lost_mean", "lost_any_seed", "held_mean", "held_any_seed",
            *(f"service_{p}" for p in PRIORITIES))
RECOMMEND, STAYS, NOT_ASSESSED = "recommend", "incumbent_stays", "not_assessed"
HOLDS = "holds"
OTHER_POLICY = "a_different_policy_cleared_the_bar"
WINNER_FAILS = "the_primary_winner_fails_a_guardrail"
NONE_CLEARED = "no_challenger_cleared_the_bar"
COLUMNS = (
    "seed", "family", "policy", "orders", "net_cents", "friction_cost_cents",
    "legitimate_orders", "legitimate_held", "legitimate_declined", "legitimate_cancelled",
    "available_minutes", "review_band", "review_minutes_used",
    *(f"reviews_{p}" for p in PRIORITIES), *(f"sla_met_{p}" for p in PRIORITIES),
)


class RuleError(ValueError):
    """The rule is malformed, or the rows cannot be judged by it."""


def _exact(value: Any, what: str) -> Fraction:
    """A YAML number as an exact fraction (a float read as the decimal it was written as)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RuleError(f"{what} must be a number, got {value!r}")
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    return Fraction(repr(float(value)))


def _count(row: Mapping[str, Any], column: str) -> int:
    value = row[column]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise RuleError(f"{column} must be a non-negative integer on seed {row.get('seed')} "
                        f"for {row.get('policy')}, got {value!r}")
    return int(value)


def _whole(value: Any, what: str) -> int:
    exact = _exact(value, what)
    if exact.denominator != 1:
        raise RuleError(f"{what} must be a whole number, got {value!r}")
    return int(exact)


def _mean(values: Mapping[int, Fraction | int]) -> Fraction:
    return sum((Fraction(v) for v in values.values()), Fraction(0)) / len(values)


def _cents(usd: Fraction) -> int:
    cents = usd * 100
    if cents.denominator != 1:
        raise RuleError(f"${float(usd)} is not a whole number of cents")
    return int(cents)


@dataclass(frozen=True)
class Cap:
    """At most ``mean`` on the mean over seeds and ``any_seed`` on every seed."""

    mean: Fraction
    any_seed: Fraction


@dataclass(frozen=True)
class Rule:
    """The rule's numbers (``reporting.recommendation_rule``) and the costs it uses."""

    lost: Cap  # per 10,000 legitimate orders
    held: Cap
    service_share: Fraction
    service_min_entries: int
    hurdle_cents: Fraction  # mean improvement per 1,000 attempted orders
    positive_seeds: Mapping[int, int]  # seeds paired -> strictly positive seeds needed
    simpler: tuple[str, ...]  # simplest first: the last tie-break
    analyst_cents_per_hour: Fraction
    ltv_cents: int  # the configured LTV proxy, the one in the rows' friction cost
    ltv_sensitivity_cents: tuple[int, ...]

    @classmethod
    def from_protocol(cls, raw: Mapping[str, Any], policy_cfg: Mapping[str, Any]) -> Rule:
        """The rule from the protocol (``raw``) and the costs from ``config/policy.yaml``.

        The loaded hourly cost is stated in both files and must agree.
        """
        try:
            spec = raw["reporting"]["recommendation_rule"]
            eligibility, hurdle = spec["eligibility"], spec["hurdle"]
            caps = {name: Cap(_exact(eligibility[key]["mean_at_most"], f"{key}.mean_at_most"),
                              _exact(eligibility[key]["any_seed_at_most"],
                                     f"{key}.any_seed_at_most"))
                    for name, key in (("lost", "lost_legitimate_per_10000"),
                                      ("held", "held_legitimate_per_10000"))}
            hourly = _exact(raw["capacity"]["analyst_loaded_hourly_usd"],
                            "capacity.analyst_loaded_hourly_usd")
            configured = _exact(policy_cfg["costs"]["analyst_loaded_hourly_usd"],
                                "costs.analyst_loaded_hourly_usd")
            ltv = _exact(policy_cfg["costs"]["false_decline_ltv_usd"],
                         "costs.false_decline_ltv_usd")
            sensitivity = [_exact(value, "sensitivity.ltv_proxy_usd")
                           for value in raw["sensitivity"]["ltv_proxy_usd"]]
            positive = dict(hurdle["positive_seeds"])
            rule = cls(
                lost=caps["lost"], held=caps["held"],
                service_share=_exact(eligibility["service_share_at_least"],
                                     "service_share_at_least"),
                service_min_entries=_whole(eligibility["service_min_pooled_entries"],
                                           "service_min_pooled_entries"),
                hurdle_cents=_exact(hurdle["mean_improvement_usd_per_1000_orders"],
                                    "mean_improvement_usd_per_1000_orders") * 100,
                positive_seeds=positive,
                simpler=tuple(spec["simpler_order"]),
                analyst_cents_per_hour=hourly * 100,
                ltv_cents=_cents(ltv),
                ltv_sensitivity_cents=tuple(_cents(value) for value in sensitivity),
            )
        except RuleError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise RuleError(f"the protocol's recommendation rule is incomplete: {error!r}") \
                from error
        if hourly != configured:
            raise RuleError(f"the protocol prices an analyst hour at ${float(hourly)}, "
                            f"config/policy.yaml at ${float(configured)}")
        rule.check()
        return rule

    def check(self) -> None:
        for name, cap in (("lost", self.lost), ("held", self.held)):
            if not 0 <= cap.mean <= cap.any_seed:
                raise RuleError(f"the {name} caps must satisfy 0 <= mean <= any seed")
        if not 0 < self.service_share <= 1 or self.service_min_entries < 1:
            raise RuleError("the service floor is a share in (0, 1] over at least one entry")
        if self.hurdle_cents < 0 or self.analyst_cents_per_hour < 0 or self.ltv_cents < 0:
            raise RuleError("the hurdle and the costs cannot be negative")
        if not self.positive_seeds:
            raise RuleError("positive_seeds lists no seed count")
        for n, needed in self.positive_seeds.items():
            if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, needed)) \
                    or not 1 <= needed <= n:
                raise RuleError(f"positive_seeds: {n}: {needed} must be whole and 1 <= needed "
                                "<= seeds")
        if len(set(self.simpler)) != len(self.simpler) or INCUMBENT not in self.simpler:
            raise RuleError("simpler_order must name each policy once, the incumbent included")
        if any(value < 0 for value in self.ltv_sensitivity_cents) or \
                len(set(self.ltv_sensitivity_cents)) != len(self.ltv_sensitivity_cents) or \
                self.ltv_cents in self.ltv_sensitivity_cents:
            raise RuleError("the LTV sensitivities must be distinct and differ from the "
                            "configured LTV proxy")

    def needed(self, seeds: int) -> int:
        """Strictly positive paired seeds a challenger needs out of ``seeds``."""
        if seeds in self.positive_seeds:
            return self.positive_seeds[seeds]
        if seeds < min(self.positive_seeds):
            return seeds
        raise RuleError(f"the rule names no sign bar for {seeds} seeds")


@dataclass(frozen=True)
class OperatingCell:
    """Where the rule is applied: a family, a replay cell and the LTV proxy.

    ``where`` maps the replay's cell columns to their values; ``capacity`` is the
    cell's name in result keys; ``ltv_cents`` is ``None`` for the configured proxy.
    ``varies`` names what differs from the primary cell (``primary`` for itself).
    """

    name: str
    varies: str
    family: str
    capacity: str
    where: Mapping[str, str]
    ltv_cents: int | None = None


@dataclass(frozen=True)
class Service:
    """One priority's queue: entries pooled over seeds and the mean share decided in time."""

    entries: int
    share: Fraction | None  # mean over the seeds with entries; None: no entries at all
    assessed: bool


@dataclass(frozen=True)
class Standing:
    """One policy in one cell, by seed: what the rule measures."""

    policy: str
    lost: Mapping[int, Fraction]  # per 10,000 legitimate orders
    held: Mapping[int, Fraction]
    net: Mapping[int, Fraction]  # rule net contribution, cents
    orders: Mapping[int, int]
    review_minutes: Mapping[int, int]
    service: Mapping[str, Service]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.net))

    def values(self) -> dict[str, Fraction | None]:
        """Each criterion's value (``None``: a priority not assessed)."""
        out: dict[str, Fraction | None] = {
            "lost_mean": _mean(self.lost), "lost_any_seed": max(self.lost.values()),
            "held_mean": _mean(self.held), "held_any_seed": max(self.held.values()),
        }
        for priority, item in self.service.items():
            out[f"service_{priority}"] = item.share if item.assessed else None
        return out


@dataclass(frozen=True)
class Verdict:
    """A challenger against the rule in one cell."""

    policy: str
    failed: tuple[str, ...]  # eligibility criteria it fails
    improvement: Mapping[int, Fraction]  # cents per 1,000 attempted orders, by paired seed
    needed: int
    clears: bool  # the hurdle

    @property
    def eligible(self) -> bool:
        return not self.failed

    @property
    def mean(self) -> Fraction | None:
        return _mean(self.improvement) if self.improvement else None

    @property
    def positive(self) -> int:
        return sum(value > 0 for value in self.improvement.values())


@dataclass(frozen=True)
class CellResult:
    """The rule's outcome in one operating cell."""

    cell: OperatingCell
    outcome: str  # RECOMMEND, STAYS or NOT_ASSESSED
    chosen: str | None  # the recommended challenger
    best: str | None  # the chosen challenger, or the best one when the incumbent stays
    incumbent_misses: tuple[str, ...]
    standings: Mapping[str, Standing]
    verdicts: Mapping[str, Verdict]
    note: str | None = None
    absent: tuple[str, ...] = ()  # policies with no feasible point on any seed here


def _bounds(rule: Rule) -> dict[str, tuple[str, Fraction]]:
    bounds = {"lost_mean": ("at_most", rule.lost.mean),
              "lost_any_seed": ("at_most", rule.lost.any_seed),
              "held_mean": ("at_most", rule.held.mean),
              "held_any_seed": ("at_most", rule.held.any_seed)}
    for priority in PRIORITIES:
        bounds[f"service_{priority}"] = ("at_least", rule.service_share)
    return bounds


def _fails(value: Fraction | None, kind: str, bound: Fraction) -> bool:
    if value is None:  # not assessed
        return False
    return value > bound if kind == "at_most" else value < bound


def standing(policy: str, rows: Sequence[Mapping[str, Any]], rule: Rule,
             ltv_cents: int | None = None) -> Standing:
    """A policy's rows in one cell (one per seed) as the rule measures them."""
    lost, held, net, orders, minutes = {}, {}, {}, {}, {}
    entries: dict[str, dict[int, int]] = {p: {} for p in PRIORITIES}
    met: dict[str, dict[int, int]] = {p: {} for p in PRIORITIES}
    ltv = rule.ltv_cents if ltv_cents is None else ltv_cents
    for row in rows:
        seed = int(row["seed"])
        if seed in net:
            raise RuleError(f"{policy} has two rows for seed {seed} in one cell")
        counts = {column: _count(row, column) for column in COLUMNS[3:] if column != "net_cents"}
        if isinstance(row["net_cents"], bool) or not isinstance(row["net_cents"],
                                                                 numbers.Integral):
            raise RuleError(f"net_cents must be an integer, got {row['net_cents']!r}")
        if counts["orders"] == 0 or counts["legitimate_orders"] == 0:
            raise RuleError(f"{policy} on seed {seed}: no orders or no legitimate orders")
        customers = counts["legitimate_declined"] + counts["legitimate_cancelled"]
        if counts["friction_cost_cents"] != rule.ltv_cents * customers:
            raise RuleError(
                f"{policy} on seed {seed}: friction_cost_cents {counts['friction_cost_cents']} "
                f"is not the configured LTV proxy ({rule.ltv_cents} cents) times the "
                f"{customers} legitimate orders declined or cancelled")
        if counts["review_band"] not in (0, 1):
            raise RuleError(f"review_band must be 0 or 1, got {counts['review_band']}")
        allotment = Fraction(0) if counts["review_band"] == 0 else \
            Fraction(counts["available_minutes"]) * rule.analyst_cents_per_hour / 60
        net[seed] = int(row["net_cents"]) - ltv * customers - allotment
        lost[seed] = Fraction(10_000 * customers, counts["legitimate_orders"])
        held[seed] = Fraction(10_000 * counts["legitimate_held"], counts["legitimate_orders"])
        orders[seed] = counts["orders"]
        minutes[seed] = counts["review_minutes_used"]
        for priority in PRIORITIES:
            entered, in_time = counts[f"reviews_{priority}"], counts[f"sla_met_{priority}"]
            if in_time > entered:
                raise RuleError(f"{policy} on seed {seed}: more {priority} reviews in time "
                                f"than entered")
            entries[priority][seed], met[priority][seed] = entered, in_time
    if not net:
        raise RuleError(f"{policy} has no rows in the cell")
    service = {}
    for priority in PRIORITIES:
        shares = [Fraction(met[priority][s], entries[priority][s])
                  for s in sorted(net) if entries[priority][s]]
        pooled = sum(entries[priority].values())
        service[priority] = Service(
            entries=pooled,
            share=sum(shares, Fraction(0)) / len(shares) if shares else None,
            assessed=pooled >= rule.service_min_entries)
    return Standing(policy, lost, held, net, orders, minutes, service)


def _rank(verdict: Verdict, item: Standing, rule: Rule) -> tuple[Any, ...]:
    """Highest mean improvement first; exact ties to fewer lost customers, then fewer
    review minutes, then the simpler policy."""
    assert verdict.mean is not None
    return (-verdict.mean, _mean(item.lost), _mean(item.review_minutes),
            rule.simpler.index(verdict.policy))


def judge(cell: OperatingCell, standings: Mapping[str, Standing], rule: Rule,
          absent: Sequence[str] = ()) -> CellResult:
    """The rule in one cell, from each evaluated policy's standing there (``absent``:
    the policies with no row in the cell)."""
    unknown = sorted((set(standings) | set(absent)) - set(rule.simpler))
    if unknown:
        raise RuleError(f"simpler_order does not rank {unknown}")
    absent = tuple(sorted(absent))
    incumbent = standings.get(INCUMBENT)
    if incumbent is None:
        return CellResult(cell, NOT_ASSESSED, None, None, (), standings, {},
                          note="the incumbent has no feasible operating point on any seed",
                          absent=absent)
    bounds = _bounds(rule)
    reference = incumbent.values()
    misses = tuple(c for c in CRITERIA if _fails(reference[c], *bounds[c]))
    for criterion in misses:  # no worse than the incumbent
        bounds[criterion] = (bounds[criterion][0], reference[criterion])
    verdicts = {}
    for policy, item in standings.items():
        if policy == INCUMBENT:
            continue
        values = item.values()
        failed = tuple(c for c in CRITERIA if _fails(values[c], *bounds[c]))
        paired = sorted(set(item.seeds) & set(incumbent.seeds))
        improvement = {}
        for seed in paired:
            if item.orders[seed] != incumbent.orders[seed]:
                raise RuleError(f"{policy} and the incumbent count different orders on seed "
                                f"{seed}")
            improvement[seed] = (item.net[seed] - incumbent.net[seed]) * 1000 \
                / item.orders[seed]
        needed = rule.needed(len(paired)) if paired else 0
        positive = sum(value > 0 for value in improvement.values())
        clears = bool(paired) and _mean(improvement) >= rule.hurdle_cents \
            and positive >= needed
        verdicts[policy] = Verdict(policy, failed, improvement, needed, clears)
    ranked = sorted((v for v in verdicts.values() if v.improvement),
                    key=lambda v: _rank(v, standings[v.policy], rule))
    candidates = [v for v in ranked if v.eligible and v.clears]
    if candidates:
        chosen = candidates[0].policy
        return CellResult(cell, RECOMMEND, chosen, chosen, misses, standings, verdicts,
                          absent=absent)
    eligible = [v for v in ranked if v.eligible]
    best = (eligible or ranked or [None])[0]
    return CellResult(cell, STAYS, None, None if best is None else best.policy, misses,
                      standings, verdicts, absent=absent)


def apply_rule(rows: Iterable[Mapping[str, Any]], cells: Sequence[OperatingCell],
               rule: Rule, policies: Sequence[str]) -> list[CellResult]:
    """The rule in each operating cell, from the replay's evaluated outcome rows (one per
    seed, family, policy and replay cell; a policy with no feasible point on a seed has
    no row there). ``policies`` are the protocol's."""
    rows = list(rows)
    columns = set().union(*(row.keys() for row in rows)) if rows else set(COLUMNS)
    if missing := sorted(set(COLUMNS) - columns):
        raise RuleError(f"the outcome rows lack columns {missing}")
    results = []
    for cell in cells:
        chosen = [row for row in rows if row["family"] == cell.family
                  and all(row[column] == value for column, value in cell.where.items())]
        by_policy: dict[str, list[Mapping[str, Any]]] = {}
        for row in chosen:
            if row["policy"] not in policies:
                raise RuleError(f"an outcome row for {row['policy']!r}, not a protocol policy")
            by_policy.setdefault(row["policy"], []).append(row)
        standings = {policy: standing(policy, items, rule, cell.ltv_cents)
                     for policy, items in sorted(by_policy.items())}
        results.append(judge(cell, standings, rule,
                             absent=[p for p in policies if p not in standings]))
    return results


def flip(primary: CellResult, other: CellResult) -> tuple[bool | None, str]:
    """Whether ``other`` reaches the primary cell's outcome, and the reason if not."""
    if NOT_ASSESSED in (primary.outcome, other.outcome):
        return None, NOT_ASSESSED
    if other.chosen == primary.chosen:
        return True, HOLDS
    winner = primary.chosen
    if winner is not None and winner in other.verdicts and other.verdicts[winner].failed:
        return False, WINNER_FAILS
    return False, (NONE_CLEARED if other.chosen is None else OTHER_POLICY)


# ---------------------------------------------------------------- result metrics and tables
def ltv_name(cents: int) -> str:
    """An LTV proxy's name in cell names and keys: ``ltv_5_usd``, ``ltv_7_50_usd``."""
    dollars = f"{cents // 100}" if cents % 100 == 0 else f"{cents // 100}_{cents % 100:02d}"
    return f"ltv_{dollars}_usd"


def _float(value: Fraction | None) -> float | None:
    return None if value is None else float(value)


def improvement_name(cell: OperatingCell) -> str:
    """The metric name of a cell's paired rule net improvement."""
    base = "rule_net_per_1000_orders"
    return base if cell.ltv_cents is None else f"{base}_{ltv_name(cell.ltv_cents)}"


def _policy_row(result: CellResult, policy: str, ltv: int) -> dict[str, Any]:
    """One policy in one cell for the ``evaluate.recommendation`` table (``None`` where
    the policy has no row in the cell, or a figure does not apply to it)."""
    cell, item = result.cell, result.standings.get(policy)
    verdict = result.verdicts.get(policy)
    values = item.values() if item is not None else dict.fromkeys(CRITERIA)
    row: dict[str, Any] = {
        "cell": cell.name, "varies": cell.varies, "family": cell.family,
        "capacity": cell.capacity, "ltv_cents": ltv, "policy": policy,
        "seeds_count": 0 if item is None else len(item.seeds),
        "lost_legitimate_per_10k_mean_bps": _float(values["lost_mean"]),
        "lost_legitimate_per_10k_max_bps": _float(values["lost_any_seed"]),
        "held_legitimate_per_10k_mean_bps": _float(values["held_mean"]),
        "held_legitimate_per_10k_max_bps": _float(values["held_any_seed"]),
    }
    for priority in PRIORITIES:
        service = None if item is None else item.service[priority]
        row[f"service_{priority}_entries_count"] = None if service is None else service.entries
        row[f"service_{priority}_in_time_mean_share"] = \
            None if service is None else _float(service.share)
        row[f"service_{priority}_assessed"] = None if service is None else service.assessed
    gains = {} if verdict is None else verdict.improvement
    row.update({
        "rule_net_vs_incumbent_rules_per_1000_mean_cents": _float(
            None if verdict is None else verdict.mean),
        "rule_net_vs_incumbent_rules_per_1000_min_cents": _float(
            min(gains.values()) if gains else None),
        "rule_net_vs_incumbent_rules_per_1000_max_cents": _float(
            max(gains.values()) if gains else None),
        "paired_seeds_count": None if verdict is None else len(gains),
        "positive_seeds_count": None if verdict is None else verdict.positive,
        "positive_seeds_needed_count": None if verdict is None else verdict.needed,
        "fails": ", ".join(result.incumbent_misses if policy == INCUMBENT
                           else () if verdict is None else verdict.failed),
        "eligible": None if verdict is None else verdict.eligible,
        "clears_hurdle": None if verdict is None else verdict.clears,
        "recommended": policy == result.chosen,
    })
    return row


def _cell_metrics(result: CellResult, ltv: int, window: str) -> dict[str, Metric]:
    cell = result.cell
    scope = f"{cell.family}.{cell.capacity}"
    metrics: dict[str, Metric] = {}
    for policy, item in result.standings.items():
        if cell.ltv_cents is None:  # the same replays as the cell at the configured proxy
            for measure, values, what in (
                    ("rule_lost_legitimate_per_10k", item.lost,
                     "legitimate orders declined at checkout, blocked, declined after "
                     "review, or cancelled after an unanswered hold, per 10,000 legitimate "
                     "orders; mean over seeds"),
                    ("rule_held_legitimate_per_10k", item.held,
                     "legitimate orders held per 10,000 legitimate orders; mean over seeds")):
                spread = SeedSpread({s: float(v) for s, v in values.items()})
                metrics[f"evaluate.{measure}.{scope}.{policy}"] = Metric(
                    value=spread.mean, unit="bps", population=what, window=window,
                    seeds=spread)
        verdict = result.verdicts.get(policy)
        if verdict is None:
            continue
        key = f"evaluate.{improvement_name(cell)}.vs_{INCUMBENT}.{scope}.{policy}"
        compared = (f"rule net contribution per 1,000 attempted orders (ledger net after the "
                    f"friction cost at an LTV proxy of ${ltv / 100:g}, less the analyst "
                    f"allotment): {policy} minus {INCUMBENT}, paired by seed")
        if verdict.improvement:
            spread = SeedSpread({s: float(v) for s, v in verdict.improvement.items()})
            metrics[key] = Metric(value=spread.mean, unit="cents", population=compared,
                                  window=window, seeds=spread)
        else:
            metrics[key] = Metric.not_evaluated(
                unit="cents", population=compared, window=window,
                reason="no seed on which both have a feasible operating point")
    return metrics


def outputs(results: Sequence[CellResult], rule: Rule, *, window: str = "test"
            ) -> tuple[dict[str, Metric], dict[str, list[dict[str, Any]]]]:
    """Metrics and tables from the cells' results; the first result is the primary cell.

    Metrics: each challenger's paired rule net improvement over the incumbent per 1,000
    attempted orders (``evaluate.rule_net_per_1000_orders[_ltv_<usd>_usd].vs_incumbent_
    rules.<family>.<capacity>.<policy>``), each policy's lost and held legitimate
    customers per 10,000 legitimate orders (mean over seeds, with the seeds), and
    ``evaluate.recommendation.holds``: the share of operating cells besides the primary
    that reach its outcome. Tables: ``evaluate.recommendation`` (a row per cell and
    policy) and ``evaluate.flips`` (a row per cell, with the reason it flips).
    """
    if not results or results[0].cell.varies != "primary" or \
            any(r.cell.varies == "primary" for r in results[1:]):
        raise RuleError("the first operating cell, and only it, must be the primary cell")
    names = [r.cell.name for r in results]
    if len(set(names)) != len(names):
        raise RuleError(f"operating cells share a name: {names}")
    metrics: dict[str, Metric] = {}
    policy_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    primary = results[0]
    holding = 0
    for result in results:
        cell = result.cell
        ltv = rule.ltv_cents if cell.ltv_cents is None else cell.ltv_cents
        for key, item in _cell_metrics(result, ltv, window).items():
            if key in metrics:
                raise RuleError(f"two operating cells publish {key}")
            metrics[key] = item
        for policy in sorted([*result.standings, *result.absent],
                             key=rule.simpler.index):
            policy_rows.append(_policy_row(result, policy, ltv))
        holds, reason = (None, "primary") if result is primary else flip(primary, result)
        holding += bool(holds)
        flip_rows.append({
            "cell": cell.name, "varies": cell.varies, "family": cell.family,
            "capacity": cell.capacity, "ltv_cents": ltv, "outcome": result.outcome,
            "recommended": result.chosen, "best_challenger": result.best,
            "incumbent_misses": ", ".join(result.incumbent_misses),
            "holds": holds, "reason": reason, "note": result.note,
        })
    assessed = [r for r in results[1:] if NOT_ASSESSED not in (r.outcome, primary.outcome)]
    population = ("operating cells besides the primary cell where the rule reaches the "
                  "primary cell's outcome (each sensitivity alone)")
    metrics["evaluate.recommendation.holds"] = (
        Metric.from_ratio(holding, len(assessed), unit="share", population=population,
                          window=window) if assessed else
        Metric.not_evaluated(unit="share", population=population, window=window,
                             numerator=0, denominator=0,
                             reason="no operating cell besides the primary was assessed"))
    return metrics, {"evaluate.recommendation": policy_rows, "evaluate.flips": flip_rows}
