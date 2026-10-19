"""
Real-world solar power forecasting with the trained ATERNET.

No solar panel is needed: measured generation comes from Sheffield Solar
PV_Live (regional, updated every 30 min) and weather from Open-Meteo's
multi-model forecasts. See config/data_source.py.

    # Forecast the next 8 hours from right now
    python predict.py live

    # Replay past forecasts and score them against measured generation
    python predict.py backtest --start 2026-10-01 --end 2026-10-16

Outputs go to outputs/predictions/ (CSV) and outputs/plots/ (PNG).
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.paths import PREDICTIONS_DIR, PLOTS_DIR, METRICS_DIR, ensure_dirs
from src.inference.forecaster import Forecaster
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger("predict")


def _plot_live(df: pd.DataFrame, region_name: str, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    t = df["target_time"]
    ax.fill_between(t, df["p10_mw"], df["p90_mw"], alpha=0.3, label="P10-P90")
    ax.plot(t, df["p50_mw"], lw=2, label="P50")
    ax.set_title(f"ATERNET live forecast - {region_name}\nissued {df['issue_time'].iloc[0]:%Y-%m-%d %H:%M} UTC")
    ax.set_ylabel("PV generation (MW)")
    ax.set_xlabel("Time (UTC)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_backtest(df: pd.DataFrame, region_name: str, lead_hours: float, path: Path):
    """Continuous trace at one lead time: each target time forecast `lead_hours` ahead."""
    d = df[df["lead_hours"] == lead_hours].sort_values("target_time")
    if d.empty:
        return
    # Reindex onto the full half-hour grid so night-time gaps between
    # issue times break the line instead of being drawn as straight joins.
    grid = pd.date_range(d["target_time"].min(), d["target_time"].max(), freq="30min")
    d = d.set_index("target_time").reindex(grid).rename_axis("target_time").reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(d["target_time"], d["p10_mw"], d["p90_mw"], alpha=0.3, label="P10-P90")
    ax.plot(d["target_time"], d["p50_mw"], ".-", ms=3, lw=1, label=f"P50 ({lead_hours:g} h ahead)")
    ax.plot(d["target_time"], d["actual_mw"], "k.", ms=4, label="Measured (PV_Live)")
    ax.set_title(f"ATERNET backtest - {region_name}")
    ax.set_ylabel("PV generation (MW)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cmd_live(args):
    fc = Forecaster(device=args.device)
    df = fc.forecast_live()
    issue = df["issue_time"].iloc[0]
    stem = f"live_forecast_{issue:%Y%m%dT%H%M}"
    df.to_csv(PREDICTIONS_DIR / f"{stem}.csv", index=False)
    _plot_live(df, fc.region.name, PLOTS_DIR / f"{stem}.png")

    print(f"\nRegion: {fc.region.name}  |  capacity {df['capacity_mw'].iloc[0]:.0f} MWp  |  "
          f"issued {issue:%Y-%m-%d %H:%M} UTC  |  regime: {df['regime_at_issue'].iloc[0]}")
    print(df[["target_time", "p10_mw", "p50_mw", "p90_mw"]].to_string(
        index=False, float_format=lambda v: f"{v:8.1f}"))
    print(f"\nSaved {PREDICTIONS_DIR / (stem + '.csv')} and {PLOTS_DIR / (stem + '.png')}")


def cmd_backtest(args):
    fc = Forecaster(device=args.device)
    hours = [int(h) for h in args.issue_hours.split(",")]
    df, metrics = fc.backtest(args.start, args.end, issue_hours=hours)
    stem = f"backtest_{args.start}_{args.end}"
    df.to_csv(PREDICTIONS_DIR / f"{stem}.csv", index=False)
    save_json(metrics, METRICS_DIR / f"{stem}.json")
    _plot_backtest(df, fc.region.name, args.plot_lead, PLOTS_DIR / f"{stem}.png")

    print(f"\nBacktest {args.start}..{args.end}  |  {fc.region.name}  |  "
          f"{metrics['n_issue_times']} forecasts, {metrics['n_steps']} half-hours scored")
    print(f"  MAE  (% of capacity)          : {metrics['nmae_pct']:.2f}")
    print(f"  RMSE (% of capacity)          : {metrics['nrmse_pct']:.2f}")
    print(f"  Daylight MAE                  : {metrics['daylight_mae_mw']:.1f} MW "
          f"(day-ahead persistence {metrics['daylight_persistence_mae_mw']:.1f} MW, "
          f"mean output {metrics['daylight_mean_actual_mw']:.1f} MW)")
    print(f"  RMSE skill vs persistence     : {metrics['rmse_skill_vs_persistence']:.3f}")
    print(f"  P10-P90 coverage (nominal 0.8): {metrics['picp']:.3f}")
    print(f"\nSaved {PREDICTIONS_DIR / (stem + '.csv')}, {METRICS_DIR / (stem + '.json')}, "
          f"{PLOTS_DIR / (stem + '.png')}")


def main():
    p = argparse.ArgumentParser(description="ATERNET real-world solar forecasting.")
    p.add_argument("--device", default="cpu")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("live", help="Forecast the next 8 hours from the latest data.")

    b = sub.add_parser("backtest", help="Score replayed forecasts against measured PV.")
    b.add_argument("--start", required=True, help="First issue day (YYYY-MM-DD, UTC).")
    b.add_argument("--end", required=True, help="Last issue day (YYYY-MM-DD, UTC).")
    b.add_argument("--issue-hours", default="4,6,8,10,12,14",
                   help="Comma-separated UTC issue hours per day.")
    b.add_argument("--plot-lead", type=float, default=2.0,
                   help="Lead time (hours) for the backtest trace plot.")

    args = p.parse_args()
    ensure_dirs()
    {"live": cmd_live, "backtest": cmd_backtest}[args.cmd](args)


if __name__ == "__main__":
    main()
