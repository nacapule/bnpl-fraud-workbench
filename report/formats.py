"""Number formats for rendered documents.

Each format turns a :class:`core.results.Metric` (or a plain value from the
protocol or a configuration file) into text. A format refuses a metric whose
unit it does not fit, so a template cannot print cents as a percentage or a
rate as dollars. Differences of rates are percentage points (``pp``), never
percent: a change from 30% to 42.5% is "+12.5 pp". A difference of two rates
is stored in ``bps`` (10,000 times the difference) or computed in the template
as ``{{ a - b | pp }}``.

A metric that was not evaluated renders as "not evaluated (N=…)" whatever the
format, so a gap is visible instead of a number.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass

from core.results import Metric
from core.stats import sign_test


class FormatError(ValueError):
    """A format does not fit the value or its arguments."""


@dataclass(frozen=True)
class Value:
    """What a placeholder resolved to: a result metric, or a plain value."""

    metric: Metric | None = None
    plain: object = None
    difference_of: str | None = None  # the unit of two metrics subtracted in a template

    @property
    def number(self) -> int | float:
        number = self.metric.value if self.metric is not None else self.plain
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise FormatError(f"{number!r} is not a number")
        if not math.isfinite(number):
            raise FormatError(f"{number!r} is not finite")
        return number


Format = Callable[[Value, list[str]], str]


def _text(number: float, decimals: int, *, signed: bool = False, prefix: str = "",
          suffix: str = "") -> str:
    """``number`` with thousands separators; no sign on a value that rounds to zero."""
    digits = f"{abs(number):,.{decimals}f}"
    zero = set(digits) <= set("0.,")
    sign = "" if zero else "-" if number < 0 else "+" if signed else ""
    return f"{sign}{prefix}{digits}{suffix}"


def _options(args: list[str], decimals: int) -> tuple[int, bool]:
    signed = False
    for arg in args:
        if arg == "signed":
            signed = True
        elif arg.isdigit():
            decimals = int(arg)
        else:
            raise FormatError(f"unknown format argument {arg!r}")
    return decimals, signed


def _unit(resolved: Value, allowed: tuple[str, ...], name: str) -> None:
    unit = resolved.metric.unit if resolved.metric is not None else resolved.difference_of
    if unit is not None and unit not in allowed:
        raise FormatError(f"format {name!r} does not fit unit {unit!r}")


def _metric(resolved: Value, name: str) -> Metric:
    if resolved.metric is None:
        raise FormatError(f"format {name!r} needs a result metric")
    return resolved.metric


def usd(resolved: Value, args: list[str]) -> str:
    """Cents as dollars: ``$1,234`` (``usd:2`` shows cents, ``usd:signed`` gives ``+$1,234``)."""
    _unit(resolved, ("cents",), "usd")
    decimals, signed = _options(args, 0)
    return _text(resolved.number / 100, decimals, signed=signed, prefix="$")


def pct(resolved: Value, args: list[str]) -> str:
    """A rate or share as percent: ``12.3%`` (one decimal by default)."""
    if resolved.difference_of is not None:
        raise FormatError("a difference of rates is in percentage points: use the pp format")
    _unit(resolved, ("rate", "share"), "pct")
    decimals, signed = _options(args, 1)
    return _text(resolved.number * 100, decimals, signed=signed, suffix="%")


def pp(resolved: Value, args: list[str]) -> str:
    """A difference of rates in percentage points, always signed: ``+12.5 pp``.

    Takes a ``bps`` metric (10,000 times the difference), or a plain number
    that is a difference of two rates (``{{ a - b | pp }}``).
    """
    if resolved.metric is not None:
        _unit(resolved, ("bps",), "pp")
        points = resolved.number / 100
    elif resolved.difference_of in ("rate", "share"):
        points = resolved.number * 100
    elif resolved.difference_of == "bps":
        points = resolved.number / 100
    else:
        raise FormatError("the pp format needs a difference of two rates")
    decimals, _ = _options(args, 1)
    return _text(points, decimals, signed=True, suffix=" pp")


def bps(resolved: Value, args: list[str]) -> str:
    """Basis points: ``12.3 bps``."""
    _unit(resolved, ("bps",), "bps")
    decimals, signed = _options(args, 1)
    return _text(resolved.number, decimals, signed=signed, suffix=" bps")


def per_10k(resolved: Value, args: list[str]) -> str:
    """A ``bps`` quantity read as a count per 10,000: ``12.3 per 10,000``."""
    _unit(resolved, ("bps",), "per_10k")
    decimals, signed = _options(args, 1)
    return _text(resolved.number, decimals, signed=signed, suffix=" per 10,000")


def count(resolved: Value, args: list[str]) -> str:
    """A whole number with thousands separators: ``1,234``."""
    _unit(resolved, ("count", "minutes"), "count")
    _, signed = _options(args, 0)
    number = resolved.number
    if isinstance(number, float) and not number.is_integer():
        raise FormatError(f"the count format needs a whole number, got {number!r}")
    return _text(number, 0, signed=signed)


def num(resolved: Value, args: list[str]) -> str:
    """A plain number with fixed decimals (two by default)."""
    decimals, signed = _options(args, 2)
    return _text(resolved.number, decimals, signed=signed)


def fraction(resolved: Value, args: list[str]) -> str:
    """Numerator over denominator: ``42/50``."""
    metric = _metric(resolved, "n")
    if args:
        raise FormatError("the n format takes no arguments")
    if metric.numerator is None:
        raise FormatError("the n format needs a metric with a numerator and denominator")
    return f"{_text(metric.numerator, 0)}/{_text(metric.denominator, 0)}"


def numerator(resolved: Value, args: list[str]) -> str:
    metric = _metric(resolved, "numerator")
    if metric.numerator is None:
        raise FormatError("this metric has no numerator")
    return _text(metric.numerator, 0)


def denominator(resolved: Value, args: list[str]) -> str:
    metric = _metric(resolved, "denominator")
    if metric.denominator is None:
        raise FormatError("this metric has no denominator")
    return _text(metric.denominator, 0)


UNIT_FORMATS: dict[str, Format] = {
    "cents": usd,
    "rate": pct,
    "share": pct,
    "bps": bps,
    "count": count,
    "minutes": count,
    "hours": num,
    "ratio": num,
    "score": num,
}


def value(resolved: Value, args: list[str]) -> str:
    """The format that fits the metric's unit; plain values as they are."""
    if resolved.metric is not None:
        return UNIT_FORMATS[resolved.metric.unit](resolved, args)
    if resolved.difference_of in ("rate", "share"):
        return pp(resolved, args)
    if resolved.difference_of is not None:
        return UNIT_FORMATS[resolved.difference_of](resolved, args)
    plain = resolved.plain
    if isinstance(plain, dt.date):
        return date(resolved, args)
    if isinstance(plain, str):
        return plain
    if isinstance(plain, int) and not isinstance(plain, bool):
        return _text(plain, 0)
    return num(resolved, args)


def _bound(number: float) -> Value:
    """A number in a metric's own unit (an interval bound, a seed's value)."""
    return Value(plain=number)


def interval(resolved: Value, args: list[str]) -> str:
    """The metric's interval in its own unit: ``95% CI 7.8% to 10.2%``."""
    metric = _metric(resolved, "ci")
    if metric.interval is None:
        raise FormatError("this metric has no interval")
    form = UNIT_FORMATS[metric.unit]
    low = form(_bound(metric.interval.low), args)
    high = form(_bound(metric.interval.high), args)
    return f"{metric.interval.level * 100:g}% CI {low} to {high}"


def seeds(resolved: Value, args: list[str]) -> str:
    """Seed replication: ``mean +$1,234, min -$200, max +$3,000; positive on 9/10 seeds``."""
    metric = _metric(resolved, "seeds")
    if metric.seeds is None:
        raise FormatError("the seeds format needs a metric with a seed spread")
    form = UNIT_FORMATS[metric.unit]
    shown = [*args, "signed"] if form is not count else args
    spread = metric.seeds

    def show(number: float) -> str:
        return form(_bound(number), shown)

    return (
        f"mean {show(spread.mean)}, min {show(spread.min)}, max {show(spread.max)}; "
        f"{signs(resolved, [])}"
    )


def signs(resolved: Value, args: list[str]) -> str:
    """Sign count over seeds: ``positive on 9/10 seeds``."""
    metric = _metric(resolved, "signs")
    if metric.seeds is None:
        raise FormatError("the signs format needs a metric with a seed spread")
    test = sign_test(metric.seeds.per_seed.values())
    if test.direction == "none":
        return f"mixed: {test.positive} positive and {test.negative} negative of {test.n} seeds"
    agreeing = test.positive if test.direction == "positive" else test.negative
    return f"{test.direction} on {agreeing}/{test.n} seeds"


def p_value(resolved: Value, args: list[str]) -> str:
    """The exact two-sided sign-test p-value over seeds: ``p = 0.021``."""
    metric = _metric(resolved, "p")
    if metric.seeds is None:
        raise FormatError("the p format needs a metric with a seed spread")
    p = sign_test(metric.seeds.per_seed.values()).p_value
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def define(resolved: Value, args: list[str]) -> str:
    """The metric's fixed definition: its population and window."""
    metric = _metric(resolved, "define")
    if metric.window == "all":
        return f"{metric.population} (all windows)"
    return f"{metric.population} ({metric.window.replace('_', '-')} window)"


def date(resolved: Value, args: list[str]) -> str:
    """A date as ``1 June 2025``."""
    plain = resolved.plain
    if isinstance(plain, str):
        try:
            plain = dt.date.fromisoformat(plain[:10])
        except ValueError as error:
            raise FormatError(f"{resolved.plain!r} is not a date") from error
    if not isinstance(plain, dt.date):
        raise FormatError(f"{resolved.plain!r} is not a date")
    return f"{plain.day} {plain:%B %Y}"


def bps_pct(resolved: Value, args: list[str]) -> str:
    """A setting in basis points as percent: 2500 becomes ``25%``."""
    decimals, _ = _options(args, 0)
    return _text(resolved.number / 100, decimals, suffix="%")


FORMATS: dict[str, Format] = {
    "value": value,
    "usd": usd,
    "pct": pct,
    "pp": pp,
    "bps": bps,
    "per_10k": per_10k,
    "count": count,
    "num": num,
    "n": fraction,
    "numerator": numerator,
    "denominator": denominator,
    "ci": interval,
    "seeds": seeds,
    "signs": signs,
    "p": p_value,
    "define": define,
    "date": date,
    "bps_pct": bps_pct,
}

# Formats that describe a metric rather than print its value, so they also
# work on a metric that was not evaluated.
DESCRIPTIVE = frozenset({"define", "n", "numerator", "denominator"})


def apply(name: str, resolved: Value, args: list[str]) -> str:
    """Format ``resolved`` with the format called ``name``."""
    if name not in FORMATS:
        raise FormatError(f"unknown format {name!r}; known: {', '.join(sorted(FORMATS))}")
    metric = resolved.metric
    if metric is not None and not metric.evaluated and name not in DESCRIPTIVE:
        if metric.denominator is None:
            return "not evaluated"
        return f"not evaluated (N={_text(metric.denominator, 0)})"
    return FORMATS[name](resolved, args)
