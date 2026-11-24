import pandas as pd

from data_generation.generator import generate_dataset, load_config
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