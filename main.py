"""DairyChain AI pipeline runner. Stage 2: generate + validate synthetic data."""
from __future__ import annotations

import sys

from data_generation.generator import generate_dataset, load_config, save_dataset
from data_generation.validator import validate


def summarize(tag: str, daily, orders) -> None:
    print(f"\n--- Scenario '{tag}' (SIMULATED DATA) ---")
    print(f"days simulated          : {len(daily)}")
    print(f"mean daily consumption  : {daily['daily_consumption'].mean():,.0f} cartons")
    print(f"orders placed           : {len(orders)}")
    print(f"mean realised lead time : {orders['lead_time_days'].mean():.1f} days")
    print(f"minimum stock           : {daily['stock_end_of_day'].min():,} cartons")
    print(f"stockout days           : {(daily['unmet_demand'] > 0).sum()}")
    print(f"total unmet demand      : {daily['unmet_demand'].sum():,} cartons")


def main() -> int:
    cfg = load_config()
    ok = True
    for tag, surge in (("normal", False), ("surge", True)):
        daily, orders = generate_dataset(cfg, surge=surge)
        report = validate(daily, orders, cfg)
        print(f"\n=== Validation: '{tag}' ===")
        for _, r in report.iterrows():
            print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['check']} {r['detail']}")
        ok = ok and bool(report["passed"].all())
        save_dataset(daily, orders, "data/synthetic", tag)
        summarize(tag, daily, orders)
    print("\nFiles written to data/synthetic/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())