from __future__ import annotations

import pandas as pd


def validate(daily: pd.DataFrame, orders: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run all checks. Returns a table with columns: check, passed, detail."""
    results = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": bool(ok), "detail": detail})

    initial = cfg["procurement"]["initial_stock_cartons"]
    moq = cfg["procurement"]["moq_cartons"]
    cap = cfg["production"]["capacity_cap_l"]

    add("no missing values", not daily.isna().any().any())
    add("dates are consecutive days",
        (daily["date"].diff().dropna() == pd.Timedelta(days=1)).all())
    add("stock never negative", (daily["stock_end_of_day"] >= 0).all())

    prev = daily["stock_end_of_day"].shift(1).fillna(initial)
    balance = prev + daily["arrivals"] - daily["daily_consumption"]
    bad = int((balance != daily["stock_end_of_day"]).sum())
    add("stock balance: stock(t) = stock(t-1) + arrivals - consumption", bad == 0,
        f"{bad} violating days")

    add("consumption + unmet = requirement",
        (daily["daily_consumption"] + daily["unmet_demand"]
         == daily["carton_requirement"]).all())
    add("unmet demand only when stock is empty",
        ((daily["unmet_demand"] == 0) | (daily["stock_end_of_day"] == 0)).all())

    got = orders.loc[orders["arrived_in_history"]].groupby("arrival_date")["order_quantity"].sum()
    expected = got.reindex(daily["date"], fill_value=0).to_numpy()
    add("arrivals match the orders table",
        (expected == daily["arrivals"].to_numpy()).all())

    pipeline = daily["orders_placed_qty"].cumsum() - daily["arrivals"].cumsum()
    add("on-order quantity is consistent",
        (pipeline == daily["on_order_end_of_day"]).all())

    add("order quantities are multiples of MOQ",
        (orders["order_quantity"] % moq == 0).all())
    add("lead times respect the minimum",
        (orders["lead_time_days"] >= cfg["procurement"]["min_lead_time_days"]).all())
    add("production within line capacity",
        ((daily["attempted_production_l"] <= cap + 1e-6)
         & (daily["actual_production_l"] <= cap + 1e-6)).all())
    add("every row is labelled synthetic",
        (daily["data_source"] == "synthetic").all()
        and (orders["data_source"] == "synthetic").all())

    return pd.DataFrame(results)