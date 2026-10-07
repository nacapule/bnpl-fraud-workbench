"""Analyst shifts, capacity and the fixed service calendar.

Times are integer seconds on the platform clock (``datetime64[s]`` as int64), which
keeps the replay's event loop cheap. Day 0 of that count (1970-01-01) is a Thursday,
so a day's weekday is ``(day + 3) % 7`` with 0 = Monday.

Capacity has two parameters (experiments/protocol.yaml): coverage hours, which the
shift *layout* sets (when someone is on shift, which decides whether a review can
finish before shipment), and review minutes, which the number of analysts per shift
sets. Utilization is measured against the analysts' summed productive minutes; the
union coverage of the shifts is reported separately (they differ whenever shifts
overlap). The SLA clock is a fixed :class:`ServiceCalendar` that does not move with the
roster, so changing the staffing never changes how service hours are counted.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from core import config

DAY = 86_400
WEEK_OFFSET = 3  # weekday of day 0 (1970-01-01, a Thursday), Monday = 0


def to_seconds(times: Any) -> np.ndarray:
    """Platform times as int64 seconds."""
    return np.asarray(times, dtype="datetime64[s]").astype(np.int64)


def weekday(day: int) -> int:
    return (day + WEEK_OFFSET) % 7


def _minutes(clock: str) -> int:
    hours, minutes = clock.split(":")
    value = int(hours) * 60 + int(minutes)
    if not 0 <= value < 24 * 60:
        raise ValueError(f"not a time of day: {clock!r}")
    return value


@dataclass(frozen=True)
class Shift:
    """A shift that starts at ``start_minute`` on each of ``days`` (it may run past midnight)."""

    name: str
    days: tuple[int, ...]
    start_minute: int
    productive_minutes: int

    def __post_init__(self) -> None:
        if not self.days or not set(self.days) <= set(range(7)):
            raise ValueError(f"shift {self.name}: days must be weekdays 0..6")
        if not 0 < self.productive_minutes <= DAY // 60:
            raise ValueError(f"shift {self.name}: productive time must be within a day")


@dataclass(frozen=True)
class Roster:
    """A shift layout and the number of analysts on each shift."""

    shifts: tuple[Shift, ...]
    analysts_per_shift: Mapping[str, int]

    def __post_init__(self) -> None:
        names = [shift.name for shift in self.shifts]
        if len(set(names)) != len(names):
            raise ValueError("shift names must be unique")
        if set(self.analysts_per_shift) != set(names):
            raise ValueError("analysts_per_shift must name every shift exactly")
        if any(not isinstance(n, int) or n < 0 for n in self.analysts_per_shift.values()):
            raise ValueError("analysts per shift must be whole, non-negative numbers")

    @classmethod
    def from_config(cls, policy: Mapping[str, Any] | None = None, *, layout: str | None = None,
                    analysts_per_shift: Mapping[str, int] | None = None) -> Roster:
        policy = config.load("policy") if policy is None else policy
        roster = policy["roster"]
        layout = layout or roster["layout"]
        shifts = tuple(
            Shift(item["name"], tuple(item["days"]), _minutes(item["start"]),
                  round(float(item["productive_hours"]) * 60))
            for item in roster["layouts"][layout]
        )
        counts = analysts_per_shift or roster["analysts_per_shift"]
        return cls(shifts, dict(counts))

    @property
    def analysts(self) -> tuple[tuple[str, Shift], ...]:
        """One entry per analyst: its name and shift."""
        return tuple((f"{shift.name}-{k + 1}", shift) for shift in self.shifts
                     for k in range(self.analysts_per_shift[shift.name]))

    def windows(self, shift: Shift, start: int, end: int) -> list[tuple[int, int]]:
        """The shift's on-shift intervals that overlap [start, end), in time order."""
        out = []
        for day in range(start // DAY - 1, end // DAY + 1):
            if weekday(day) in shift.days:
                begin = day * DAY + shift.start_minute * 60
                finish = begin + shift.productive_minutes * 60
                if finish > start and begin < end:
                    out.append((begin, finish))
        return out

    def available_minutes(self, start: int, end: int) -> float:
        """Summed productive analyst-minutes inside [start, end)."""
        total = 0
        for _, shift in self.analysts:
            total += sum(min(b, end) - max(a, start) for a, b in self.windows(shift, start, end))
        return total / 60

    def coverage_minutes(self, start: int, end: int) -> float:
        """Minutes inside [start, end) with at least one analyst on shift (union)."""
        spans = sorted((max(a, start), min(b, end)) for shift in self.shifts
                       if self.analysts_per_shift[shift.name]
                       for a, b in self.windows(shift, start, end))
        total, reach = 0, start
        for a, b in spans:
            if b > reach:
                total += b - max(a, reach)
                reach = b
        return total / 60


class AnalystClock:
    """One analyst's on-shift time: when the next review can start and when it ends.

    Work happens only on shift. A review that reaches the end of a shift resumes at
    the start of the analyst's next shift, so busy time inside any shift never exceeds
    that shift's productive minutes.
    """

    def __init__(self, windows: list[tuple[int, int]]) -> None:
        self.starts = [a for a, _ in windows]
        self.ends = [b for _, b in windows]
        if any(b <= a for a, b in windows) or any(
                self.starts[i + 1] < self.ends[i] for i in range(len(windows) - 1)):
            raise ValueError("an analyst's shifts must be ordered and must not overlap")

    def next_on_shift(self, t: int) -> int | None:
        """The earliest instant at or after ``t`` on shift (None after the last shift)."""
        i = bisect_right(self.ends, t)
        if i >= len(self.starts):
            return None
        return max(t, self.starts[i])

    def finish(self, start: int, seconds: int) -> int | None:
        """When work of ``seconds`` begun at ``start`` (on shift) completes."""
        i = bisect_right(self.ends, start)
        t, left = start, seconds
        while i < len(self.starts):
            t = max(t, self.starts[i])
            room = self.ends[i] - t
            if left <= room:
                return t + left
            left -= room
            i += 1
        return None


@dataclass(frozen=True)
class ServiceCalendar:
    """Fixed service hours (the SLA clock), the same whatever the roster."""

    days: tuple[int, ...]
    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        if not 0 <= self.start_minute < self.end_minute <= 24 * 60:
            raise ValueError("service hours must start and end within one day")
        if not set(self.days) <= set(range(7)):
            raise ValueError("service days must be weekdays 0..6")

    @classmethod
    def from_config(cls, policy: Mapping[str, Any] | None = None) -> ServiceCalendar:
        policy = config.load("policy") if policy is None else policy
        cal = policy["sla"]["calendar"]
        end = 24 * 60 if cal["end"] == "24:00" else _minutes(cal["end"])
        return cls(tuple(cal["days"]), _minutes(cal["start"]), end)

    def _before(self, t: np.ndarray) -> np.ndarray:
        """Service seconds from time 0 to each ``t``."""
        t = np.asarray(t, dtype=np.int64)
        open_s, close_s = self.start_minute * 60, self.end_minute * 60
        pattern = np.array([close_s - open_s if weekday(m) in self.days else 0
                            for m in range(7)], dtype=np.int64)
        prefix = np.concatenate([[0], np.cumsum(pattern)])
        day, offset = np.divmod(t, DAY)
        weeks, rest = np.divmod(day, 7)
        full = weeks * prefix[7] + prefix[rest]
        partial = np.clip(offset - open_s, 0, close_s - open_s) * (pattern[rest] > 0)
        return full + partial

    def service_hours(self, t0: Any, t1: Any) -> np.ndarray:
        """Service hours between each pair of times (zero when ``t1 <= t0``)."""
        a = np.asarray(t0, dtype=np.int64)
        b = np.asarray(t1, dtype=np.int64)
        return np.maximum(self._before(b) - self._before(a), 0) / 3600
