"""
Layer 1 - simulated demand for the factory's products (customer orders), in liters/day.

SIMULATED. Not real market statistics. Effects multiply (additive in log space):
  log D(t) = log(base) + trend + weekday + season + calendar events + AR(1) shock + noise
             + random unannounced shocks (+ optional scenario surge)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_generation.calendar_events import event_log_effects


def generate_market_demand(cfg: dict, dates: pd.DatetimeIndex, rng, surge: bool = False,
                           day0: int = 0):
    """
    Return (demand_l array, components DataFrame). `dates` may start before the simulation start
    (warm-up); `day0` is the index of the first real simulation day inside `dates`.
    """
    m = cfg["market_demand"]
    n = len(dates)
    t = np.arange(n)

    trend = np.log1p(m["annual_trend"]) * t / 365.25
    weekday = np.log(np.array(m["weekday_factors"], dtype=float)[np.asarray(dates.day_of_week)])
    doy = np.asarray(dates.day_of_year)
    season = np.log1p(m["season_amplitude"]
                      * np.cos(2 * np.pi * (doy - m["season_peak_day_of_year"]) / 365.25))

    events = event_log_effects(dates, m["events"])
    ev_total = np.sum(list(events.values()), axis=0) if events else np.zeros(n)

    ar = np.zeros(n)
    eta = rng.normal(0, m["ar_sigma"], n)
    for i in range(1, n):
        ar[i] = m["ar_phi"] * ar[i - 1] + eta[i]
    noise = rng.normal(0, m["noise_sigma"], n)

    shocks = np.zeros(n)
    sh = m["shocks"]
    starts = np.where(rng.random(n) < sh["probability_per_day"])[0]
    for s in starts:
        dur = int(rng.integers(sh["duration_days"][0], sh["duration_days"][1] + 1))
        shocks[s:s + dur] += rng.normal(0, sh["magnitude_sigma"])

    surge_log = np.zeros(n)
    if surge:
        sc = cfg["scenarios"]
        a = day0 + sc["surge_start_day"]
        surge_log[a: a + sc["surge_duration_days"]] = np.log(sc["surge_multiplier"])

    log_d = (np.log(m["base_daily_demand_l"]) + trend + weekday + season + ev_total
             + ar + noise + shocks + surge_log)
    comp = pd.DataFrame({"trend": trend, "weekday": weekday, "season": season,
                         "events": ev_total, "ar": ar, "noise": noise, "shocks": shocks,
                         "surge": surge_log}, index=dates)
    return np.exp(log_d), comp