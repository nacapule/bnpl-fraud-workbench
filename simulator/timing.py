"""Clock: one time-of-day profile and one seasonal calendar for every actor.

Times are integer seconds since the Unix epoch (naive platform-local time)
until the world is assembled. Every human action, legitimate or fraudulent,
takes its time of day from the same profile; legitimate shopping also follows
the seasonal calendar (a holiday peak and busier weekends). Fraud episodes
start evenly over the horizon and then follow the same hours.

Times are drawn inside their allowed interval from the profile restricted to
that interval, never drawn freely and then clipped, so no time piles up on an
interval's boundary.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

DAY = 86_400
HOUR = 3_600
MINUTE = 60

# Share of activity by hour of day (evening peak), the same for every actor.
HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 8, 9, 10, 12, 12, 10, 8, 5, 3], dtype=float
)
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
# Cumulative profile mass at each hour boundary of a day (0 at midnight, 1 at the next).
_KNOTS_S = np.arange(25, dtype=float) * HOUR
_KNOTS_F = np.concatenate([[0.0], np.cumsum(HOUR_P)])
_KNOTS_F[-1] = 1.0
HOLIDAY = ((11, 15), (12, 23))  # inclusive month-day range of the holiday peak


def seconds(ts: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000_000)


def _mass(second_of_day: np.ndarray | float) -> np.ndarray:
    """Profile mass between midnight and a second of the day (0 to 86,400)."""
    return np.interp(second_of_day, _KNOTS_S, _KNOTS_F)


def _second(mass: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`_mass`: the second of the day at a cumulative mass."""
    return np.interp(mass, _KNOTS_F, _KNOTS_S)


_KS = _KNOTS_S.tolist()
_KF = _KNOTS_F.tolist()
_HP = HOUR_P.tolist()


def _mass1(second_of_day: int) -> float:
    hour = min(second_of_day // HOUR, 23)
    return _KF[hour] + _HP[hour] * (second_of_day - hour * HOUR) / HOUR


def _within_day(rng: np.random.Generator, day: int, a: int, b: int) -> int:
    """Scalar :func:`_within_days`."""
    mass = _mass1(a) + rng.random() * (_mass1(b) - _mass1(a))
    hour = min(bisect.bisect_right(_KF, mass) - 1, 23)
    offset = int(hour * HOUR + (mass - _KF[hour]) / _HP[hour] * HOUR)
    return day + min(max(offset, a), max(a, b - 1))


def _within_days(rng: np.random.Generator, day: np.ndarray, a: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
    """A time on each ``day`` (midnight) in seconds ``[a, b)`` of the day, by the profile."""
    lo, hi = _mass(a), _mass(b)
    offset = np.floor(_second(lo + rng.random(len(day)) * (hi - lo))).astype(np.int64)
    return day + np.minimum(np.maximum(offset, a), np.maximum(a, b - 1))


class Clock:
    """Calendar of the order horizon with the seasonal shopping intensity."""

    def __init__(self, order_start: pd.Timestamp, order_end: pd.Timestamp,
                 holiday_lift: float, weekend_lift: float) -> None:
        self.start = seconds(order_start)
        self.end = seconds(order_end)
        days = pd.date_range(order_start, order_end, freq="D", inclusive="left")
        month_day = days.month * 100 + days.day
        lo, hi = HOLIDAY
        holiday = (month_day >= lo[0] * 100 + lo[1]) & (month_day <= hi[0] * 100 + hi[1])
        weight = np.ones(len(days))
        weight[holiday] *= holiday_lift
        weight[days.dayofweek >= 5] *= weekend_lift
        self.day_weight = weight
        self.cum = np.concatenate([[0.0], np.cumsum(weight)])
        self.n_days = len(days)

    # ---------------------------------------------------------- calendar
    def exposure(self, lo: int, hi: int) -> float:
        """Seasonal weight between two instants (in day units)."""
        lo, hi = max(lo, self.start), min(hi, self.end)
        if hi <= lo:
            return 0.0
        return self._cum_at(hi) - self._cum_at(lo)

    def _cum_at(self, t: int) -> float:
        offset = (t - self.start) / DAY
        day = min(int(offset), self.n_days - 1)
        return float(self.cum[day] + (offset - day) * self.day_weight[day])

    def seasonal_times(self, rng: np.random.Generator, n: int, lo: int, hi: int) -> np.ndarray:
        """``n`` sorted shopping times in ``[lo, hi)``: days by seasonal weight, the time
        of day by the profile within the part of the day inside the interval."""
        lo, hi = max(lo, self.start), min(hi, self.end)
        if n == 0 or hi <= lo:
            return np.zeros(0, dtype=np.int64)
        a, b = self._cum_at(lo), self._cum_at(hi)
        targets = a + rng.random(n) * (b - a)
        days = np.clip(np.searchsorted(self.cum, targets, side="right") - 1, 0, self.n_days - 1)
        midnight = self.start + days.astype(np.int64) * DAY
        first = np.maximum(lo - midnight, 0)
        last = np.minimum(hi - midnight, DAY)
        return np.sort(_within_days(rng, midnight, first, last))

    # ------------------------------------------------------- time of day
    def between(self, rng: np.random.Generator, lo: int, hi: int) -> int:
        """A time in ``[lo, hi)`` drawn from the time-of-day profile over that interval."""
        if hi <= lo + 1:
            return lo
        d0, d1 = lo - lo % DAY, (hi - 1) - (hi - 1) % DAY
        if d0 == d1:
            day, a, b = d0, lo - d0, hi - d0
        else:
            head = 1.0 - _mass1(lo - d0)
            tail = _mass1(hi - d1)
            middle = (d1 - d0) // DAY - 1
            u = rng.random() * (head + middle + tail)
            if u < head:
                day, a, b = d0, lo - d0, DAY
            elif u < head + middle:
                day, a, b = d0 + (1 + min(int(u - head), middle - 1)) * DAY, 0, DAY
            else:
                day, a, b = d1, 0, hi - d1
        return _within_day(rng, day, a, b)

    def around(self, rng: np.random.Generator, target: int, after: int | None = None) -> int:
        """A profile time within half a day of ``target`` (later than ``after``)."""
        lo = target - DAY // 2
        if after is not None:
            lo = max(lo, after + 1)
        return self.between(rng, lo, lo + DAY)

    def after(self, rng: np.random.Generator, t: int, lo_days: float, hi_days: float) -> int:
        """A profile time between ``lo_days`` and ``hi_days`` after ``t``."""
        lo = t + max(1, int(lo_days * DAY))
        return self.between(rng, lo, max(lo + HOUR, t + int(hi_days * DAY)))

    def stratified_starts(self, rng: np.random.Generator, n: int, lo: int, hi: int) -> list[int]:
        """``n`` episode starts spread evenly over ``[lo, hi)``: one in each of ``n``
        equal slices, drawn by the profile within the slice."""
        if n <= 0:
            return []
        width = (hi - lo) / n
        return [self.between(rng, lo + int(k * width), lo + int((k + 1) * width))
                for k in range(n)]


def session_gap(rng: np.random.Generator, mean_minutes: float) -> int:
    """Seconds to the next action in one sitting (at least one minute)."""
    return int(MINUTE + rng.exponential(mean_minutes * MINUTE))
