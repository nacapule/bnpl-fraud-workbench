"""Clock: one time-of-day profile and one seasonal calendar for every actor.

Times are integer seconds since the Unix epoch (naive platform-local time)
until the world is assembled. Every human action, legitimate or fraudulent,
takes its hour of day from the same profile (F25); legitimate shopping also
follows the seasonal calendar (a holiday peak and busier weekends). Fraud
episodes start uniformly over the horizon and then follow the same hours.
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
HOUR_CDF = np.cumsum(HOUR_P)
_HOUR_CDF = HOUR_CDF.tolist()
HOLIDAY = ((11, 15), (12, 23))  # inclusive month-day range of the holiday peak


def seconds(ts: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000_000)


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
    def day_index(self, t: int) -> int:
        return (t - self.start) // DAY

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
        """``n`` sorted shopping times in ``[lo, hi)``: days by seasonal weight, hours by
        profile."""
        lo, hi = max(lo, self.start), min(hi, self.end)
        if n == 0 or hi <= lo:
            return np.zeros(0, dtype=np.int64)
        a, b = self._cum_at(lo), self._cum_at(hi)
        targets = a + rng.random(n) * (b - a)
        days = np.searchsorted(self.cum, targets, side="right") - 1
        days = np.clip(days, 0, self.n_days - 1)
        times = self.start + days.astype(np.int64) * DAY + self.hour_offsets(rng, n)
        times = np.clip(times, lo, hi - 1)
        return np.sort(times)

    # ------------------------------------------------------- time of day
    @staticmethod
    def hour_offsets(rng: np.random.Generator, n: int) -> np.ndarray:
        hours = np.minimum(np.searchsorted(HOUR_CDF, rng.random(n), side="right"), 23)
        return hours.astype(np.int64) * HOUR + rng.integers(0, HOUR, n)

    @staticmethod
    def hour_offset(rng: np.random.Generator) -> int:
        hour = min(bisect.bisect_right(_HOUR_CDF, rng.random()), 23)
        return hour * HOUR + int(rng.integers(0, HOUR))

    def at_profile_hour(self, rng: np.random.Generator, t: int) -> int:
        """The same day as ``t``, at an hour drawn from the profile."""
        return t - (t % DAY) + self.hour_offset(rng)

    def after(self, rng: np.random.Generator, t: int, lo_days: float, hi_days: float) -> int:
        """A profile-hour time between about ``lo_days`` and ``hi_days`` after ``t``
        (always later than ``t``)."""
        target = t + int(rng.uniform(lo_days, hi_days) * DAY)
        out = self.at_profile_hour(rng, target)
        while out <= t:
            out += DAY
        return out

    def uniform_start(self, rng: np.random.Generator, lo: int, hi: int) -> int:
        """A day uniformly in ``[lo, hi)`` at a profile hour (episode starts)."""
        day = lo + int(rng.integers(0, max(1, (hi - lo) // DAY))) * DAY
        out = self.at_profile_hour(rng, day)
        return min(max(out, lo), hi - 1)

    def stratified_starts(self, rng: np.random.Generator, n: int, lo: int, hi: int) -> list[int]:
        """``n`` episode starts spread evenly over ``[lo, hi)``: one in each of ``n``
        equal slices, at a uniform day within the slice and a profile hour."""
        if n <= 0:
            return []
        width = (hi - lo) / n
        starts = []
        for k in range(n):
            a = lo + int(k * width)
            b = lo + int((k + 1) * width)
            starts.append(self.uniform_start(rng, a, max(b, a + DAY)))
        return starts


def session_gap(rng: np.random.Generator, mean_minutes: float) -> int:
    """Seconds to the next action in one sitting (at least one minute)."""
    return int(MINUTE + rng.exponential(mean_minutes * MINUTE))
