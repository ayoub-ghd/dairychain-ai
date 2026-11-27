"""
DairyChain AI - synthetic data generator (v2: demand -> planning -> production -> cartons).

EVERYTHING PRODUCED HERE IS SIMULATED. It is not real industrial or market data.

Layers, simulated in this order:
  1. market demand (liters/day)       demand.py    trend, weekday, season, events, hidden shocks
  2. production plan (liters/day)     planning.py  planner sees only the past + public calendar
  3. actual production                plan x execution noise, limited by line capacity and carton stock
  4. packaging consumption (cartons)  liters / carton volume x (1 + waste)
  5. inventory and ordering           stock(t) = stock(t-1) + arrivals - consumption; weekly review
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_generation.demand import generate_market_demand
from data_generation.planning import planner_forecast, production_plan, raw_milk_series


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dataset(cfg: dict, surge: bool = False, seed: int | None = None):
    """Return (daily_df, orders_df). All data is synthetic."""
    seed = cfg["project"]["random_seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    n = cfg["simulation"]["n_days"]
    warm = cfg["planning"]["warmup_days"]
    prod, prc = cfg["production"], cfg["procurement"]
    hp, pol = cfg["history_policy"], cfg["policy"]
    vol, waste = cfg["product"]["carton_volume_l"], cfg["product"]["waste_rate"]
    moq = prc["moq_cartons"]

    start = pd.Timestamp(cfg["simulation"]["start_date"])
    all_dates = pd.date_range(start - pd.Timedelta(days=warm), periods=n + warm, freq="D")
    dates = all_dates[warm:]

    # ---- Layer 1: demand (with a warm-up period so the planner has history on day 0)
    demand_all, comp = generate_market_demand(cfg, all_dates, rng, surge=surge, day0=warm)
    # ---- Layer 2: planner (28 days ahead, past demand only), limited by capacity and milk
    forecast_all = planner_forecast(demand_all, all_dates, comp["events"].to_numpy(),
                                    comp["surge"].to_numpy(), cfg)
    demand_l, forecast_l = demand_all[warm:], forecast_all[warm:]
    surge_flag = comp["surge"].to_numpy()[warm:] != 0
    raw_milk = raw_milk_series(cfg, dates, rng)
    milk_available = raw_milk * cfg["raw_milk"]["liquid_share"]
    plan = production_plan(forecast_l, milk_available, cfg)
    # ---- Layer 3: what the line tries to run, and the cartons that requires
    attempted = np.clip(plan * (1 + rng.normal(0, prod["execution_noise_std"], n)),
                        0, prod["capacity_cap_l"])
    requirement = np.round(attempted / vol * (1 + waste)).astype(int)

    mdm = cfg["market_demand"]
    expected_daily = (mdm["base_daily_demand_l"] * np.mean(mdm["weekday_factors"])
                      * (1 + waste) / vol)

    # ---- Layers 4-5: carton stock, consumption, weekly ordering
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
            "milk_available_l": round(float(milk_available[t]), 1),
            "market_demand_l": round(float(demand_l[t]), 1),
            "planner_forecast_l": round(float(forecast_l[t]), 1),
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