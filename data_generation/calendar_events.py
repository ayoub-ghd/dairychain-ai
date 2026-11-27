"""
Calendar events for the demand simulation.

Two different things live here, on purpose:
  * event_log_effects(...)  : the TRUE (hidden) effect of each event on demand -> used by the generator only.
  * calendar_features(...)  : PUBLIC information (dates of events) -> the only thing a forecaster may use.
The forecaster learns the effect SHAPES from data; it is never given the effect sizes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
        

def event_log_effects(dates: pd.DatetimeIndex, events_cfg: dict) -> dict[str, np.ndarray]:
    """log(1 + effect) per day and per event (inside the window, before it, after it)."""
    d = dates.normalize()
    out = {}
    for name, e in events_cfg.items():
        eff = np.zeros(len(d))
        for start, end in e["windows"]:
            s, t = pd.Timestamp(start), pd.Timestamp(end)
            inside = np.asarray((d >= s) & (d <= t))
            pre = np.asarray((d >= s - pd.Timedelta(days=e["pre_days"])) & (d < s))
            post = np.asarray((d > t) & (d <= t + pd.Timedelta(days=e["post_days"])))
            eff = (eff + inside * np.log1p(e["effect"]) + pre * np.log1p(e["pre_effect"])
                   + post * np.log1p(e["post_effect"]))
        out[name] = eff
    return out


def calendar_features(dates: pd.DatetimeIndex, events_cfg: dict, cap: int = 30) -> pd.DataFrame:
    """
    Public calendar features per event GROUP ('observance', 'holiday'):
      in_<g>    : 1 if the day is inside an event window
      to_<g>    : days until the next window starts (0 inside a window), capped
      since_<g> : days since the last window ended (0 inside a window), capped
    Only the DATES are used, never the effect sizes or the pre/post lengths.
    """
    d = np.asarray(dates.normalize(), dtype="datetime64[D]")
    groups: dict[str, list] = {}
    for e in events_cfg.values():
        groups.setdefault(e["group"], []).extend(
            (np.datetime64(s, "D"), np.datetime64(t, "D")) for s, t in e["windows"])
    feats = {}
    for g, wins in groups.items():
        wins = sorted(wins)
        starts = np.array([w[0] for w in wins])
        ends = np.array([w[1] for w in wins])
        inside = np.zeros(len(d), dtype=bool)
        for s, t in wins:
            inside |= (d >= s) & (d <= t)
        to = np.full(len(d), cap)
        since = np.full(len(d), cap)
        for i, day in enumerate(d):
            nxt = starts[starts >= day]
            if len(nxt):
                to[i] = min(cap, int((nxt[0] - day) / np.timedelta64(1, "D")))
            prv = ends[ends < day]
            if len(prv):
                since[i] = min(cap, int((day - prv[-1]) / np.timedelta64(1, "D")))
        to[inside] = 0
        since[inside] = 0
        feats[f"in_{g}"] = inside.astype(int)
        feats[f"to_{g}"] = to
        feats[f"since_{g}"] = since
    return pd.DataFrame(feats, index=dates)