"""Interactive overview of a synthetic scenario. Run: python -m data_generation.plot_overview normal"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generation.generator import load_dataset


def build_figure(daily, tag: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Carton requirement (cartons/day)",
                        "Stock at end of day (cartons) and deliveries",
                        "Production (liters/day)"),
    )
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["carton_requirement"],
                             name="daily requirement", opacity=0.35), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["date"],
                             y=daily["carton_requirement"].rolling(7).mean(),
                             name="7-day mean"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["stock_end_of_day"],
                             name="stock"), row=2, col=1)
    arr = daily[daily["arrivals"] > 0]
    fig.add_trace(go.Scatter(x=arr["date"], y=arr["stock_end_of_day"], mode="markers",
                             name="delivery day", marker=dict(size=9, symbol="triangle-up")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["planned_production_l"],
                             name="planned"), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["actual_production_l"],
                             name="actual", opacity=0.6), row=3, col=1)
    fig.update_layout(height=900,
                      title=f"SYNTHETIC DATA - scenario '{tag}' (simulated, not real industrial data)")
    return fig


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "normal"
    daily, _ = load_dataset("data/synthetic", tag)
    Path("outputs").mkdir(exist_ok=True)
    path = Path("outputs") / f"synthetic_overview_{tag}.html"
    build_figure(daily, tag).write_html(path, auto_open=True)
    print("Saved", path)