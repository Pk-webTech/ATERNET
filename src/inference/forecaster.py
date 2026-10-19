"""
Live forecasting and out-of-sample backtesting with the trained ATERNET.

Both paths run the *same* code from raw API data to a P10/P50/P90 power
forecast, so a good backtest is evidence about the live forecaster
rather than about a separate evaluation pipeline:

    PV_Live history + Open-Meteo weather
      -> pv_processor.clean_pv_series / nwp_processor.compute_ensemble_stats
      -> feature_engineering.build_features
      -> create_windows.build_inference_window
      -> train-fit scalers -> ATERNET -> quantiles x capacity

`forecast_live` uses the current Open-Meteo forecast run and the latest
PV_Live readings. `backtest` replays past issue times using only data at
or before each issue time (PV is cut at the issue time before cleaning),
then scores against what PV_Live actually measured.

Caveat for backtests: the Open-Meteo historical-forecast archive stitches
together the first hours of successive model runs, so its "forecast" for
the next 8 h is somewhat more accurate than a single live run would be.
Backtest errors are therefore mildly optimistic relative to live use.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from config.model_config import DATA_CONFIG
from config.paths import BEST_CHECKPOINT_PATH, SCALERS_PATH, RAW_METADATA_DIR
from src.evaluation.metrics import all_metrics
from src.models.aternet import ATERNET
from src.phase1.create_windows import build_inference_window
from src.phase1.feature_engineering import build_features
from src.phase1.nwp_processor import compute_ensemble_stats
from src.phase1.pv_processor import clean_pv_series
from src.phase1.pvlive_openmeteo import (
    fetch_pvlive, fetch_weather, weather_to_pv_grid, half_hour_grid,
)
from src.phase1.regions import Region, get_region
from src.training.checkpoint import load_checkpoint
from src.utils.constants import ID_TO_REGIME
from src.utils.feature_groups import REGIME_ID_INDEX, NWP_FEATURES
from src.utils.helpers import minmax_transform
from src.utils.io import load_json, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)

STEP = pd.Timedelta(minutes=DATA_CONFIG.freq_minutes)
# Lookback (24 h) + 3 h for the rolling-volatility features + slack for
# PV_Live gaps that cleaning interpolates across.
HISTORY = pd.Timedelta(days=3)


class Forecaster:
    def __init__(self, checkpoint_path=BEST_CHECKPOINT_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, payload = load_checkpoint(checkpoint_path, model_class=ATERNET,
                                              map_location=self.device)
        self.model.to(self.device).eval()
        self.checkpoint_epoch = payload.get("epoch")
        self.scalers = load_pickle(SCALERS_PATH)
        self.site = load_json(RAW_METADATA_DIR / "resolved_site_metadata.json")
        if self.site.get("source") != "pvlive_openmeteo":
            raise RuntimeError(
                "The trained model was not built from the PV_Live + Open-Meteo source "
                "(config/data_source.py), so it has no live data feed. Set "
                "DATA_SOURCE = 'pvlive_openmeteo', delete data/raw/*, and rebuild + retrain."
            )
        self.horizon = DATA_CONFIG.horizon
        self.window_size = DATA_CONFIG.window_size

    @property
    def region(self) -> Region:
        return get_region(int(self.site["region_id"]))

    # ------------------------------------------------------------------
    # Raw data -> windows
    # ------------------------------------------------------------------

    def _window_at(self, pv_raw: pd.DataFrame, nwp_stats: pd.DataFrame,
                   issue_time: pd.Timestamp) -> dict:
        """Build the model input for one issue time using data <= issue_time only."""
        pv_hist = pv_raw[(pv_raw["timestamp"] <= issue_time)
                         & (pv_raw["timestamp"] > issue_time - HISTORY)]
        if pv_hist.empty or pv_hist["timestamp"].max() != issue_time:
            raise ValueError(f"No PV_Live reading at issue time {issue_time}.")

        pv_clean = clean_pv_series(pv_hist)
        hist_nwp = nwp_stats[nwp_stats["timestamp"] <= issue_time]
        feats = build_features(pv_clean, hist_nwp)
        future = nwp_stats[nwp_stats["timestamp"] > issue_time]
        window = build_inference_window(feats, future[["timestamp"] + NWP_FEATURES])
        window["capacity_kw"] = float(pv_clean["capacity_kw"].iloc[-1])
        return window

    @staticmethod
    def _nwp_stats(pv_raw: pd.DataFrame, weather_hourly: pd.DataFrame,
                   end: pd.Timestamp) -> pd.DataFrame:
        grid = half_hour_grid(pv_raw["timestamp"].min(), end)
        return compute_ensemble_stats(weather_to_pv_grid(weather_hourly, grid))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _scale(self, arr: np.ndarray, key: str) -> np.ndarray:
        shape = arr.shape
        return minmax_transform(arr.reshape(-1, shape[-1]), self.scalers[key]).reshape(shape)

    @torch.no_grad()
    def predict_windows(self, windows: list) -> dict:
        x_seq = np.concatenate([w["X_seq"] for w in windows])
        x_sigma = np.concatenate([w["X_sigma"] for w in windows])
        x_nwp = np.concatenate([w["X_nwp"] for w in windows])
        to_t = lambda a: torch.from_numpy(a.astype(np.float32)).to(self.device)
        out = self.model(to_t(self._scale(x_seq, "X_seq")),
                         to_t(self._scale(x_nwp, "X_nwp")),
                         to_t(self._scale(x_sigma, "X_sigma")))
        res = {k: out[k].cpu().numpy().clip(0.0, 1.0) for k in ("p10", "p50", "p90")}
        res["gate_weights"] = out["gate_weights"].cpu().numpy()
        res["sigma_total"] = out["sigma_total"].cpu().numpy().reshape(len(windows), -1).mean(axis=1)
        return res

    def _to_frame(self, windows: list, preds: dict) -> pd.DataFrame:
        rows = []
        for i, w in enumerate(windows):
            cap_mw = w["capacity_kw"] / 1000.0
            regime = ID_TO_REGIME[int(w["meta"][0, REGIME_ID_INDEX])]
            for h, t in enumerate(w["target_times"]):
                rows.append({
                    "issue_time": w["issue_time"],
                    "target_time": t,
                    "lead_hours": (h + 1) * STEP.total_seconds() / 3600.0,
                    "regime_at_issue": regime,
                    "capacity_mw": cap_mw,
                    "p10_mw": preds["p10"][i, h] * cap_mw,
                    "p50_mw": preds["p50"][i, h] * cap_mw,
                    "p90_mw": preds["p90"][i, h] * cap_mw,
                    "p10_norm": preds["p10"][i, h],
                    "p50_norm": preds["p50"][i, h],
                    "p90_norm": preds["p90"][i, h],
                    "sigma_total": preds["sigma_total"][i],
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def forecast_live(self, return_history: bool = False):
        """
        Forecast the next `horizon` half-hours from the latest PV_Live reading.
        With `return_history=True` also returns the measured PV used as input.
        """
        region = self.region
        now = pd.Timestamp.now(tz="UTC")
        pv_raw = fetch_pvlive(region, now - HISTORY - pd.Timedelta(hours=6), now, cache=False)
        pv_raw = pv_raw.dropna(subset=["pv_power_kw"])
        issue_time = pv_raw["timestamp"].max()

        weather = fetch_weather(region, live=True, past_days=4, forecast_days=2, cache=False)
        nwp_stats = self._nwp_stats(pv_raw, weather, issue_time + self.horizon * STEP)

        window = self._window_at(pv_raw, nwp_stats, issue_time)
        df = self._to_frame([window], self.predict_windows([window]))
        logger.info(f"Live forecast for {region.name}: issued {issue_time} "
                    f"(capacity {window['capacity_kw'] / 1000:.0f} MWp), "
                    f"peak P50 {df['p50_mw'].max():.1f} MW.")
        return (df, pv_raw) if return_history else df

    def backtest(self, start, end, issue_hours: Iterable[int] = (4, 6, 8, 10, 12, 14)) -> tuple:
        """
        Replay forecasts issued at `issue_hours` (UTC) on every day in
        [start, end] and score them against measured PV_Live generation.
        Returns (per-step DataFrame, metrics dict).
        """
        region = self.region
        start = pd.Timestamp(start, tz="UTC")
        end = pd.Timestamp(end, tz="UTC")
        data_end = end + pd.Timedelta(days=1)
        # Only cache API responses for ranges PV_Live will no longer revise.
        settled = data_end < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)

        pv_raw = fetch_pvlive(region, start - HISTORY - pd.Timedelta(hours=6), data_end,
                              cache=settled)
        weather = fetch_weather(region, start - pd.Timedelta(days=4),
                                data_end + pd.Timedelta(days=1), live=False, cache=settled)
        nwp_stats = self._nwp_stats(pv_raw, weather, data_end)

        actual = pv_raw.dropna(subset=["pv_power_kw"]).set_index("timestamp")
        windows, skipped = [], []
        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            for hour in issue_hours:
                issue = day + pd.Timedelta(hours=hour)
                try:
                    w = self._window_at(pv_raw, nwp_stats, issue)
                except ValueError as e:
                    skipped.append((issue, str(e)))
                    continue
                if not w["target_times"].isin(actual.index).all():
                    skipped.append((issue, "PV_Live actuals missing in horizon"))
                    continue
                windows.append(w)

        if skipped:
            logger.warning(f"Backtest skipped {len(skipped)} issue times, e.g. {skipped[0]}.")
        if not windows:
            raise RuntimeError("No backtest windows could be built for the requested range.")

        df = self._to_frame(windows, self.predict_windows(windows))
        df["actual_mw"] = actual.loc[df["target_time"], "pv_power_kw"].to_numpy() / 1000.0
        df["actual_norm"] = df["actual_mw"] / df["capacity_mw"]
        # Day-ahead persistence: same half-hour yesterday -- the standard
        # naive reference for solar, robust to the diurnal cycle.
        yday = df["target_time"] - pd.Timedelta(days=1)
        df["persistence_norm"] = (actual["pv_power_kw"].reindex(yday).to_numpy() / 1000.0
                                  / df["capacity_mw"].to_numpy())

        metrics = self._score(df, len(windows))
        return df, metrics

    @staticmethod
    def _score(df: pd.DataFrame, n_windows: int) -> dict:
        ok = df["persistence_norm"].notna()
        d = df[ok]
        m = all_metrics(d["actual_norm"].to_numpy(), d["p10_norm"].to_numpy(),
                        d["p50_norm"].to_numpy(), d["p90_norm"].to_numpy(),
                        y_baseline=d["persistence_norm"].to_numpy())
        daylight = d[(d["actual_norm"] > 0.01) | (d["p50_norm"] > 0.01)]
        err_mw = daylight["p50_mw"] - daylight["actual_mw"]
        pers_err = (daylight["persistence_norm"] - daylight["actual_norm"]) * daylight["capacity_mw"]
        m.update({
            "n_issue_times": int(n_windows),
            "n_steps": int(len(d)),
            "daylight_mae_mw": float(err_mw.abs().mean()),
            "daylight_rmse_mw": float(np.sqrt((err_mw ** 2).mean())),
            "daylight_persistence_mae_mw": float(pers_err.abs().mean()),
            "daylight_mean_actual_mw": float(daylight["actual_mw"].mean()),
        })
        return m
