"""Shifts, the service calendar and keyed draws, on hand-computable toys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queue_sim.draws import keyed_uniform, lognormal
from queue_sim.reviewer import service_seconds
from queue_sim.roster import AnalystClock, Roster, ServiceCalendar, Shift, to_seconds

T = pd.Timestamp


def s(stamp: str) -> int:
    return int(to_seconds(T(stamp)))


def test_weekdays_follow_the_calendar() -> None:
    monday = s("2025-06-02 00:00") // 86_400
    roster = Roster((Shift("mon", (0,), 9 * 60, 60),), {"mon": 1})
    windows = roster.windows(roster.shifts[0], s("2025-06-01"), s("2025-06-09"))
    assert windows == [(s("2025-06-02 09:00"), s("2025-06-02 10:00"))]
    assert monday * 86_400 == s("2025-06-02")


def test_a_review_crossing_the_end_of_a_shift_carries_only_what_is_left() -> None:
    roster = Roster((Shift("day", (0, 1), 9 * 60, 8 * 60),), {"day": 1})
    clock = AnalystClock(roster.windows(roster.shifts[0], s("2025-06-01"), s("2025-06-04")))
    assert clock.finish(s("2025-06-02 16:55"), 600) == s("2025-06-03 09:05")
    assert clock.next_on_shift(s("2025-06-02 17:00")) == s("2025-06-03 09:00")
    assert clock.next_on_shift(s("2025-06-03 18:00")) is None  # no shift after Tuesday
    assert clock.finish(s("2025-06-03 16:55"), 600) is None


def test_an_overnight_shift_belongs_to_the_day_it_starts() -> None:
    roster = Roster((Shift("night", (4,), 22 * 60, 8 * 60),), {"night": 1})  # Fridays
    windows = roster.windows(roster.shifts[0], s("2025-06-01"), s("2025-06-15"))
    assert windows == [(s("2025-06-06 22:00"), s("2025-06-07 06:00")),
                       (s("2025-06-13 22:00"), s("2025-06-14 06:00"))]
    assert roster.coverage_minutes(s("2025-06-07 00:00"), s("2025-06-08 00:00")) == 360


def test_capacity_is_summed_minutes_and_coverage_is_counted_once() -> None:
    """Four analysts on one shift give four times the minutes, not four times the hours.

    Measuring offered work against union coverage reported 1.21 where the team was
    loaded to 0.62 of its minutes."""
    roster = Roster((Shift("day", tuple(range(7)), 9 * 60, 8 * 60),), {"day": 4})
    week = (s("2025-06-02"), s("2025-06-09"))
    assert roster.coverage_minutes(*week) == 7 * 8 * 60
    assert roster.available_minutes(*week) == 4 * 7 * 8 * 60
    offered = 0.62 * roster.available_minutes(*week)
    union_ratio = offered / roster.coverage_minutes(*week)  # the misleading denominator
    assert union_ratio == pytest.approx(2.48)
    assert offered / roster.available_minutes(*week) == pytest.approx(0.62)


def test_overlapping_shifts_cover_their_union() -> None:
    roster = Roster((Shift("a", (0,), 9 * 60, 8 * 60), Shift("b", (0,), 13 * 60, 5 * 60)),
                    {"a": 1, "b": 1})
    day = (s("2025-06-02"), s("2025-06-03"))
    assert roster.coverage_minutes(*day) == 9 * 60
    assert roster.available_minutes(*day) == 13 * 60


def test_the_service_calendar_is_fixed_and_counts_only_service_hours() -> None:
    weekdays = ServiceCalendar((0, 1, 2, 3, 4), 8 * 60, 20 * 60)
    friday_evening, monday_morning = s("2025-06-06 19:00"), s("2025-06-09 09:00")
    assert weekdays.service_hours(friday_evening, monday_morning) == pytest.approx(2.0)
    daily = ServiceCalendar(tuple(range(7)), 8 * 60, 20 * 60)
    assert daily.service_hours(friday_evening, monday_morning) == pytest.approx(1 + 24 + 1)
    stamps = np.array([s("2025-06-02 07:00"), s("2025-06-02 12:00")])
    assert np.allclose(daily.service_hours(stamps, stamps + 3600), [0.0, 1.0])
    assert daily.service_hours(monday_morning, friday_evening) == 0.0


def test_keyed_draws_depend_only_on_seed_stream_and_key() -> None:
    keys = np.arange(1, 2001)
    first = keyed_uniform(416, "review-service", keys)
    assert np.array_equal(first[::-1], keyed_uniform(416, "review-service", keys[::-1]))
    assert keyed_uniform(416, "review-service", 7)[0] == first[6]
    assert not np.array_equal(first, keyed_uniform(416, "check-outcome-contact", keys))
    assert not np.array_equal(first, keyed_uniform(417, "review-service", keys))
    assert 0.0 <= first.min() and first.max() < 1.0
    assert abs(first.mean() - 0.5) < 0.02


def test_review_times_have_the_configured_mean() -> None:
    seconds = service_seconds(416, np.arange(200_000), 7.0, 0.6)
    assert abs(seconds.mean() / 60 - 7.0) < 0.05
    median = lognormal(np.array([0.5]), median=6.0, sigma=1.0)
    assert median[0] == pytest.approx(6.0)
