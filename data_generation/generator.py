"""
DairyChain AI - synthetic data generator.

EVERYTHING PRODUCED HERE IS SIMULATED. It is not real industrial data.

Causal chain simulated day by day:
  production plan -> attempted production -> carton requirement
  -> consumption (limited by stock) -> stock update (+arrivals -consumption)
  -> weekly review -> order placed -> arrives after a sampled lead time.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _production_series(cfg: dict, dates: pd.DatetimeIndex, rng, surge: bool):
    """Planned production (liters) per day, plus seasonal factor and surge flag."""
    p = cfg["production"]
    n = len(dates)
    weekday = np.array(p["weekday_factors"])[dates.day_of_week.to_numpy()]
    doy = dates.day_of_year.to_numpy()
    seasonal = 1 + p["seasonal_amplitude"] * np.sin(2 * np.pi * (doy - 90) / 365)
    plan_noise = 1 + rng.normal(0, p["plan_noise_std"], n)
    plan = p["base_daily_plan_l"] * weekday * seasonal * plan_noise

    surge_flag = np.zeros(n, dtype=bool)
    if surge:
        s = cfg["scenarios"]
        start = s["surge_start_day"]
        surge_flag[start:start + s["surge_duration_days"]] = True
        plan = np.where(surge_flag, plan * s["surge_multiplier"], plan)

    plan = np.clip(plan, 0, p["capacity_cap_l"])
    return plan, seasonal, surge_flag


def generate_dataset(cfg: dict, surge: bool = False, seed: int | None = None):
    """Return (daily_df, orders_df). All data is synthetic."""
    seed = cfg["project"]["random_seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    n = cfg["simulation"]["n_days"]
    prod, prc = cfg["production"], cfg["procurement"]
    hp, pol = cfg["history_policy"], cfg["policy"]
    vol, waste = cfg["product"]["carton_volume_l"], cfg["product"]["waste_rate"]
    moq = prc["moq_cartons"]

    dates = pd.date_range(cfg["simulation"]["start_date"], periods=n, freq="D")
    plan, seasonal, surge_flag = _production_series(cfg, dates, rng, surge)

    # Raw milk intake: contractual, continuous, seasonal (no weekday pattern).
    raw_milk = prod["base_daily_plan_l"] * seasonal * (1 + rng.normal(0, 0.03, n))

    # What the line tries to run, and the cartons that requires.
    attempted = np.clip(
        plan * (1 + rng.normal(0, prod["execution_noise_std"], n)),
        0, prod["capacity_cap_l"],
    )
    requirement = np.round(attempted / vol * (1 + waste)).astype(int)

    expected_daily = (prod["base_daily_plan_l"] * np.mean(prod["weekday_factors"])
                      * (1 + waste) / vol)

    stock = int(prc["initial_stock_cartons"])
    on_order = 0
    arrivals_by_day: dict[int, int] = {}
    consumed_history: list[int] = []
    hold_until, hold_start = None, 0
    rows, orders = [], []

    for t in range(n):
        # 1) deliveries arriving today
        arrived = arrivals_by_day.pop(t, 0)
        stock += arrived
        on_order -= arrived

        # 2) consumption, limited by available stock
        need = int(requirement[t])
        used = min(need, stock)
        stock -= used
        unmet = need - used
        consumed_history.append(used)

        # 3) weekly review: place an order if inventory position is too low
        placed = 0
        if t % pol["review_period_days"] == 0:
            window = consumed_history[-hp["demand_estimate_window_days"]:]
            avg = float(np.mean(window)) if len(window) >= 7 else expected_daily
            position = stock + on_order
            if position < hp["reorder_point_days_cover"] * avg:
                place = True
                if hold_until is None and rng.random() < hp["late_order_probability"]:
                    lo, hi = hp["late_order_delay_days"]
                    hold_until, hold_start = t + int(rng.integers(lo, hi + 1)), t
                    place = False                      # simulated late ordering
                elif hold_until is not None and t < hold_until:
                    place = False
                if place:
                    raw_qty = hp["order_up_to_days_cover"] * avg - position
                    qty = int(math.ceil(raw_qty / moq) * moq)
                    lead = int(max(prc["min_lead_time_days"],
                                   round(rng.normal(prc["lead_time_mean_days"],
                                                    prc["lead_time_std_days"]))))
                    arrival_day = t + lead
                    if arrival_day < n:
                        arrivals_by_day[arrival_day] = arrivals_by_day.get(arrival_day, 0) + qty
                    on_order += qty
                    placed = qty
                    orders.append({
                        "order_id": f"PO-{len(orders) + 1:04d}",
                        "order_date": dates[t],
                        "order_quantity": qty,
                        "lead_time_days": lead,
                        "arrival_date": dates[t] + pd.Timedelta(days=lead),
                        "arrived_in_history": arrival_day < n,
                        "order_delay_days": 0 if hold_until is None else t - hold_start,
                        "data_source": "synthetic",
                    })
                    hold_until = None
            else:
                hold_until = None

        rows.append({
            "date": dates[t],
            "raw_milk_received_l": round(float(raw_milk[t]), 1),
            "planned_production_l": round(float(plan[t]), 1),
            "attempted_production_l": round(float(attempted[t]), 1),
            "actual_production_l": round(used / (1 + waste) * vol, 1),
            "carton_requirement": need,
            "daily_consumption": used,
            "unmet_demand": unmet,
            "arrivals": arrived,
            "orders_placed_qty": placed,
            "on_order_end_of_day": on_order,
            "stock_end_of_day": stock,
            "demand_surge_flag": bool(surge_flag[t]),
            "data_source": "synthetic",
        })

    return pd.DataFrame(rows), pd.DataFrame(orders)


def save_dataset(daily, orders, out_dir: str, tag: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / f"daily_{tag}.csv", index=False)
    orders.to_csv(out / f"orders_{tag}.csv", index=False)


def load_dataset(out_dir: str, tag: str):
    out = Path(out_dir)
    daily = pd.read_csv(out / f"daily_{tag}.csv", parse_dates=["date"])
    orders = pd.read_csv(out / f"orders_{tag}.csv",
                         parse_dates=["order_date", "arrival_date"])
    return daily, orders