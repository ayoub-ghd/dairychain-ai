"""
Layer 2 - production planner. It works `plan_horizon_days` (28) ahead and sees ONLY past demand.

forecast(t) = mean demand on the same weekday over the `history_weeks` weeks ending >= 28 days before t
              x exp( event_knowledge * (calendar effect on t - calendar effect on the reference days)
                     + (scheduled surge on t - on the reference days) )
plan(t)     = min( forecast * (1 + margin), line capacity, liquid milk available on t )

The planner is imperfect by design: it lags seasonality/trend, knows only a share of each event's
true effect, and cannot see unannounced shocks. A scheduled surge is a commercial decision, so the
planner knows it (but only 28 days ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def planner_forecast(demand, dates: pd.DatetimeIndex, ev_total, surge_log, cfg: dict) -> np.ndarray:
    m, pl = cfg["market_demand"], cfg["planning"]
    lead = cfg["production"]["plan_horizon_days"]
    weeks, kappa = pl["history_weeks"], pl["event_knowledge"]
    demand = np.asarray(demand, dtype=float)
    n = len(demand)
    prior = m["base_daily_demand_l"] * np.array(m["weekday_factors"])[np.asarray(dates.day_of_week)]
    fc = np.empty(n)
    for t in range(n):
        idx = [t - lead - 7 * j for j in range(weeks)]
        if min(idx) < 0:                         # not enough history yet: planner's own prior
            fc[t] = prior[t] * np.exp(kappa * ev_total[t] + surge_log[t])
            continue
        ref = demand[idx].mean()
        adj = kappa * (ev_total[t] - np.mean(ev_total[idx])) + (surge_log[t] - np.mean(surge_log[idx]))
        fc[t] = ref * np.exp(adj)
    return fc


def raw_milk_series(cfg: dict, dates: pd.DatetimeIndex, rng) -> np.ndarray:
    r = cfg["raw_milk"]
    doy = np.asarray(dates.day_of_year)
    seasonal = 1 + r["seasonal_amplitude"] * np.cos(2 * np.pi * (doy - r["peak_day_of_year"]) / 365.25)
    return r["base_daily_intake_l"] * seasonal * (1 + rng.normal(0, r["noise_std"], len(dates)))


def production_plan(forecast, milk_available, cfg: dict) -> np.ndarray:
    margin = cfg["planning"]["safety_margin"]
    cap = cfg["production"]["capacity_cap_l"]
    return np.clip(np.minimum(np.asarray(forecast) * (1 + margin), np.minimum(cap, milk_available)), 0, None)