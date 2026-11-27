import numpy as np
import pandas as pd

from data_generation.calendar_events import calendar_features
from data_generation.generator import generate_dataset, load_config
from data_generation.planning import planner_forecast
from data_generation.validator import validate


def _cfg():
    return load_config("config.yaml")


def test_all_validation_checks_pass_normal_and_surge():
    cfg = _cfg()
    for surge in (False, True):
        daily, orders = generate_dataset(cfg, surge=surge)
        report = validate(daily, orders, cfg)
        assert report["passed"].all(), report[~report["passed"]]


def test_same_seed_gives_same_data():
    cfg = _cfg()
    a, _ = generate_dataset(cfg)
    b, _ = generate_dataset(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_surge_increases_carton_requirement_in_window():
    cfg = _cfg()
    s = cfg["scenarios"]
    lo, hi = s["surge_start_day"], s["surge_start_day"] + s["surge_duration_days"]
    normal, _ = generate_dataset(cfg, surge=False)
    surge, _ = generate_dataset(cfg, surge=True)
    assert (surge["carton_requirement"].iloc[lo:hi].sum()
            > normal["carton_requirement"].iloc[lo:hi].sum())
    assert surge["demand_surge_flag"].sum() == s["surge_duration_days"]


def test_validator_catches_a_tampered_stock_value():
    cfg = _cfg()
    daily, orders = generate_dataset(cfg)
    daily.loc[100, "stock_end_of_day"] += 12345
    assert not validate(daily, orders, cfg)["passed"].all()


def test_planner_never_sees_demand_after_the_planning_date():
    """plan(t) is made 28 days ahead: changing demand on days > t-28 must not change forecast(t)."""
    cfg = _cfg()
    lead = cfg["production"]["plan_horizon_days"]
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    rng = np.random.default_rng(0)
    demand = 45000 + rng.normal(0, 3000, 300)
    zeros = np.zeros(300)
    base = planner_forecast(demand, dates, zeros, zeros, cfg)
    t = 200
    changed = demand.copy()
    changed[t - lead + 1:] *= 3.0                    # everything the planner must NOT see
    new = planner_forecast(changed, dates, zeros, zeros, cfg)
    assert new[t] == base[t]


def test_calendar_features_use_dates_only():
    cfg = _cfg()
    dates = pd.date_range("2024-03-01", "2024-04-30", freq="D")
    f = calendar_features(dates, cfg["market_demand"]["events"])
    assert f.loc["2024-03-20", "in_observance"] == 1
    assert f.loc["2024-03-01", "to_observance"] == 10
    assert f.loc["2024-04-15", "since_observance"] == 6