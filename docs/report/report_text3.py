"""Chapters 6-9 and Appendix A."""

import ast
from pathlib import Path

from report_data import ROOT, FIG, d, fmt, js, M
from report_lib import C_, J, L_, R_

A = d.A


def _row_nums(r, nd_e=2):
    return [f"{r['nrmse_pct']['mean']:.{nd_e}f} ± {r['nrmse_pct']['std']:.{nd_e}f}",
            f"{r['nmae_pct']['mean']:.{nd_e}f} ± {r['nmae_pct']['std']:.{nd_e}f}",
            f"{r['r2']['mean']:.4f}", f"{r['picp']['mean']:.3f}", f"{r['pinaw']['mean']:.4f}",
            f"{r['winkler']['mean']:.4f} ± {r['winkler']['std']:.4f}"]


def chapter6(R, shots):
    R.chapter_heading(6, "Project Demonstration")
    order = ["overview", "better", "india", "sure", "safety", "explore", "experts"]
    labels, n = {}, R.fig_count.get(6, 0)
    for k in order:
        labels[k] = [f"Figure 6.{n + i + 1}" for i in range(len(shots[k]))]
        n += len(shots[k])

    def ref(k):
        ls = labels[k]
        return ls[0] if len(ls) == 1 else f"{ls[0]} to {ls[-1].split()[-1]}"

    def show(k, caption):
        for i, path in enumerate(shots[k]):
            cap = caption if len(shots[k]) == 1 else f"{caption} (part {i + 1} of {len(shots[k])})"
            R.fig(path, cap, width_cm=15.4, max_h_cm=21.0)

    R.body("The results of the project are presented in an interactive web dashboard written with Streamlit [@streamlit]. "
           "It is started with the single command **streamlit run app.py** and opens in a browser at "
           "http://localhost:8501. The left menu has nine pages; each opens with a short blue box that says in "
           "plain language what the page shows, and a collapsible glossary defines every technical term "
           "(typical error, range hit-rate, range quality score and so on). All screenshots in this chapter "
           "were captured from the running application.")
    R.table(
        ["Page", "What it shows"],
        [
            ["Overview", "Headline numbers, a colour-coded scoreboard against other models, an example forecast and results by weather situation"],
            ["Is ATERNET better?", "The multi-seed comparison for two regions, charts, calibration and what each model part contributes"],
            ["India demo (simulated)", "VIT Vellore forecasts on a clearly marked simulated plant, with a forecast viewer"],
            ["Live forecast", "Forecast of the next 8 hours from the latest data and replay of past forecasts"],
            ["Explore forecasts", "Browse individual test cases by weather situation"],
            ["How the experts work", "How much the model trusts each of the four experts"],
            ["How sure is the model?", "Relation between the model's unsureness, the range width and the real error"],
            ["Built-in safety checks", "Live demonstration of the three structural guarantees"],
            ["Training history", "Learning curves, expert usage during training, learning rate"],
        ],
        "Pages of the dashboard.", widths=[4.4, 11.2], size=9.5, align=[L_, L_])
    R.h2("6.1", "Overview Page")
    R.body(f"The overview ({ref('overview')}) opens with five headline numbers, each with a tooltip that gives the "
           "technical name. Below them, a green or yellow summary and a scoreboard compare ATERNET with LightGBM, "
           "PatchTST and LSTM for both regions. A pill is green when ATERNET is better, yellow when the result is "
           "about the same, red when it is worse and grey when the other model is unreliable. The first part of the "
           "figure also shows the navigation menu.")
    show("overview", "Overview page of the dashboard (screenshot)")
    R.h2("6.2", "Is ATERNET Better?")
    R.body(f"This page ({ref('better')}) is the key one for evaluation. A region is chosen at the top; the short "
           "answer, the scoreboard and two bar charts follow. A difference is called 'better' or 'worse' only when it "
           "exceeds about twice the run-to-run variation; otherwise it is 'about the same'. Further down, the page "
           "shows whether the forecast range holds what it promises and which parts of the model matter.")
    show("better", "Comparison page for the Yorkshire region (screenshot)")
    R.h2("6.3", "India Demonstration (Simulated)")
    R.body(f"The India page ({ref('india')}) begins with a red banner stating that the data are simulated and "
           "explaining how. It then shows the results for the simulated VIT Vellore plant, a scoreboard against "
           "LightGBM, an interactive forecast viewer in which the test case is chosen with a slider, and the "
           "diagnostic plots.")
    show("india", "India demonstration page with the simulated-data warning (screenshot)")
    R.h2("6.4", "Uncertainty, Safety Checks and Further Pages")
    R.body(f"{ref('sure')} shows the page on which the model's 'unsureness' score and the width of its forecast "
           "range are compared with the real error of each forecast, together with the calibration table. "
           f"{ref('safety')} shows the safety checks: moving the slider changes the unsureness fed to the decoder "
           "and gate, and the page reports that the range never flips (zero violations), widens monotonically and "
           "that the experts' votes spread out.")
    show("sure", "Page showing how the model's unsureness relates to the real error (screenshot)")
    show("safety", "Built-in safety checks page (screenshot)")
    R.body(f"{ref('explore')} shows the forecast explorer, where individual test cases can be browsed by weather "
           f"situation, and {ref('experts')} the page on how the four experts are used.")
    show("explore", "Forecast explorer page (screenshot)")
    show("experts", "Page showing how the four experts are trusted (screenshot)")
    R.h2("6.5", "Running the Demonstration")
    R.body("To reproduce the demonstration from the project folder:")
    R.code("python -m pytest -q                  # 74 tests pass\n"
           "python build_results_summary.py     # refresh docs/results_summary.json\n"
           "streamlit run app.py                # open http://localhost:8501\n"
           "python predict.py live              # optional: forecast the next 8 hours")


def chapter7(R):
    y, sw = d.Y[A], d.SW[A]
    ylg, ypt, yls = d.Y["LightGBM quantile"], d.Y["PatchTST quantile"], d.Y["LSTM quantile"]
    slg, spt = d.SW["LightGBM quantile"], d.SW["PatchTST quantile"]
    ynoema = d.Y["ATERNET without EMA"]
    cy, cs, cv = d.cal["yorkshire"], d.cal["south_west"], d.cal["vellore"]
    hl = d.hl

    R.chapter_heading(7, "Result and Discussion (Cost Analysis as Applicable)")
    R.h2("7.1", "Experimental Setup")
    R.body("All experiments use the datasets of Table 5.1, the models and settings of Chapter 5 and the evaluation "
           "metrics of Table 5.6. Unless stated otherwise, a result is the mean and standard deviation of three "
           "independent training runs (seeds 42, 43 and 44); the LSTM, whose runs differ greatly, was run six times. "
           "Every value is computed on the held-out test period (8 March to 29 September 2026), which was not used "
           "for training, early stopping or calibration. A difference between two models is called significant only "
           "if it exceeds about twice the larger run-to-run standard deviation. A single training run takes about "
           "ten minutes on the laptop described in Section 3.3.1.")

    R.h2("7.2", "Results on Yorkshire")
    R.body("Table 7.1 and Figure 7.1 compare ATERNET with the three baselines on the Yorkshire region.")
    ordered = [("ATERNET (final)", y), ("ATERNET without EMA", ynoema), ("LightGBM", ylg), ("PatchTST", ypt), ("LSTM (6 runs)", yls)]
    R.table(["Model", "nRMSE %", "nMAE %", "R²", "PICP (all)", "PINAW", "Winkler"],
            [[n] + _row_nums(r) for n, r in ordered],
            "Yorkshire test results, mean ± standard deviation over training runs (lower nRMSE, nMAE, PINAW and "
            "Winkler are better). PICP counts night points; see Section 7.5.",
            widths=[3.4, 2.4, 2.2, 1.7, 1.7, 1.8, 2.4], size=8.5, align=[L_, C_, C_, C_, C_, C_, C_], highlight_rows=(0,))
    R.fig(FIG / "fig_7_1_comparison_yorkshire.png", "Yorkshire: accuracy and quality of the forecast range for ATERNET and the baselines.", width_cm=15.2)
    R.body(f"The final ATERNET reaches an nRMSE of **{fmt(y, 'nrmse_pct', 2)}%**. LightGBM scores "
           f"{fmt(ylg, 'nrmse_pct', 2)}%; the 0.02 gap is about one standard deviation of ATERNET, so the two are "
           f"**equally accurate** on this region (LightGBM has a slightly lower mean absolute error, "
           f"{ylg['nmae_pct']['mean']:.2f}% against {y['nmae_pct']['mean']:.2f}%, and the R² values are equal). "
           f"ATERNET is clearly better than PatchTST ({fmt(ypt, 'nrmse_pct', 2)}%; ATERNET's error is about 11% lower). The Winkler score of ATERNET ({fmt(y, 'winkler', 4)}) is lower than that of "
           f"LightGBM ({fmt(ylg, 'winkler', 4)}) and PatchTST ({fmt(ypt, 'winkler', 4)}), and its intervals are "
           f"narrower (PINAW {y['pinaw']['mean']:.4f} against {ylg['pinaw']['mean']:.4f} and "
           f"{ypt['pinaw']['mean']:.4f}). The narrower ranges partly reflect that ATERNET's raw ranges under-cover in "
           "daylight (Section 7.5), and the daylight coverage of the baselines was not measured, so the Winkler score, "
           "which penalises missed values, is the fairer comparison. The spread between ATERNET's three runs is very "
           f"small ({y['nrmse_pct']['std']:.3f}), which also shows that training is stable.")
    pers = hl["baseline_rmse"] * 100
    R.body(f"The single main model (seed 42) used by the dashboard and the backtest reaches nRMSE "
           f"{hl['nrmse_pct']:.2f}%, nMAE {hl['nmae_pct']:.2f}% and R² {hl['r2']:.3f}. Persistence (assuming "
           f"the next hours equal the present) has an nRMSE of {pers:.1f}%, so ATERNET cuts the error by "
           f"{hl['rmse_skill_vs_persistence'] * 100:.0f}%.")
    seeds_lstm = yls["per_seed_nrmse"]
    R.body("**The LSTM baseline is unstable.** Three of its six runs ended at 7.0 to 9.2% nRMSE, and the other three at "
           f"{min(seeds_lstm):.2f} to {sorted(seeds_lstm)[2]:.2f}%. Table 7.2 lists every run. On its good runs the "
           "LSTM is about as accurate as ATERNET and two of the three had a lower Winkler score (0.086 to 0.090 against "
           "0.092), so we do not claim that ATERNET beats a "
           "well-trained LSTM; we claim only that ATERNET delivers its accuracy on every run, which the LSTM does "
           "not. We did not investigate why the LSTM fails on some seeds.")
    rows2 = []
    for k, seed in enumerate((42, 43, 44, 45, 46, 47)):
        def g(r):
            v = r["per_seed_nrmse"]
            return f"{v[k]:.2f}" if k < len(v) else "–"
        rows2.append([str(seed), g(y), g(ylg), g(ypt), g(yls)])
    R.table(["Seed", "ATERNET", "LightGBM", "PatchTST", "LSTM"], rows2,
            "Test nRMSE (%) of every training run on Yorkshire ('–' = not run).",
            widths=[2.0, 3.2, 3.2, 3.2, 3.2], size=9.5, align=[C_, C_, C_, C_, C_])

    R.h2("7.3", "Results on South West England")
    R.body("To test whether the result depends on Yorkshire, the whole pipeline was re-run on the PV_Live region of "
           "South West England using a separate workspace. ATERNET used exactly the same settings as in Yorkshire; "
           "the baselines used their defaults. The LSTM and the ablations were not run on this region.")
    R.table(["Model", "nRMSE %", "nMAE %", "R²", "PICP (all)", "PINAW", "Winkler"],
            [["ATERNET (final)"] + _row_nums(sw, 2), ["LightGBM"] + _row_nums(slg, 2), ["PatchTST"] + _row_nums(spt, 2)],
            "South West England test results over three training runs.",
            widths=[3.4, 2.4, 2.2, 1.7, 1.7, 1.8, 2.4], size=8.5, align=[L_, C_, C_, C_, C_, C_, C_], highlight_rows=(0,))
    R.fig(FIG / "fig_7_2_comparison_south_west.png", "South West England: accuracy and quality of the forecast range.", width_cm=15.2)
    R.body(f"Here ATERNET is more accurate than both baselines: nRMSE {fmt(sw, 'nrmse_pct', 2)}% against "
           f"{fmt(slg, 'nrmse_pct', 2)}% for LightGBM and {fmt(spt, 'nrmse_pct', 2)}% for PatchTST. The gap to "
           f"LightGBM (about 2%) is several times the standard deviations, so it is a real but modest advantage. "
           f"The Winkler score is also lower ({sw['winkler']['mean']:.4f} against {slg['winkler']['mean']:.4f} and "
           f"{spt['winkler']['mean']:.4f}). The mean absolute errors of ATERNET and LightGBM are equal "
           f"({sw['nmae_pct']['mean']:.2f}% and {slg['nmae_pct']['mean']:.2f}%), so ATERNET's advantage lies in the "
           "larger errors that dominate the RMSE. Because ATERNET's settings were chosen on Yorkshire and not adjusted, "
           "this region is an out-of-sample check of the design.")

    R.h2("7.4", "Development of the Model and Ablation Studies")
    R.body("The final model is the result of a sequence of changes, each tested over three seeds (Table 7.4 and "
           "Figure 7.3).")
    R.table(["Configuration", "Runs", "nRMSE %", "Winkler", "PICP (all)"],
            [[n, str(k), f"{m:.2f} ± {s:.2f}", f"{wm:.4f}", f"{pm:.3f}"] for n, k, m, s, wm, ws, pm, ps in d.dev],
            "Development of the model on Yorkshire (three seeds each).",
            widths=[6.6, 1.3, 3.0, 2.3, 2.4], size=9.5, align=[L_, C_, C_, C_, C_], highlight_rows=(4,))
    R.fig(FIG / "fig_7_3_model_development.png", "nRMSE of each stage of the development (points = mean, whiskers = standard deviation; dashed line = LightGBM).", width_cm=14.0)
    R.bullets([
        f"**The learned gate hurt.** With the original gate the error was {d.dev[0][2]:.2f} ± {d.dev[0][3]:.2f}%, worse "
        f"and far more variable than simply averaging the four experts ({d.dev[1][2]:.2f} ± {d.dev[1][3]:.3f}%). "
        "Every run with the learned gate was worse than every run with uniform averaging.",
        f"**The residual gate removed the harm** ({d.dev[2][2]:.2f} ± {d.dev[2][3]:.2f}%). It starts as the uniform "
        "average and can only move away from it if that lowers the validation loss. It matched, but did not beat, "
        "uniform averaging.",
        f"**The last-step shortcut** lowered the error to {d.dev[3][2]:.2f}%. In the earlier model the gap to LightGBM was largest at "
        "the first forecast steps, which showed that the model was not using the latest observation well; the "
        "shortcut gives it a direct linear path from that observation.",
        f"**Weight averaging (EMA)** lowered the error to {d.dev[4][2]:.2f}% and reduced the run-to-run spread about "
        "five-fold.",
    ])
    R.body("To find out which parts of the final design contribute, we switched them off one at a time (Table 7.5 and "
           "Figure 7.4). These ablations were trained without EMA and are compared with the same configuration "
           "without EMA.")
    ref = ynoema
    abl_rows, hl_rows = [], []
    for r in [r for r in R_ABL()]:
        gap = r["nrmse_pct"]["mean"] - ref["nrmse_pct"]["mean"]
        tol = 2 * max(r["nrmse_pct"]["std"], ref["nrmse_pct"]["std"])
        wgap = r["winkler"]["mean"] - ref["winkler"]["mean"]
        wtol = 2 * max(r["winkler"]["std"], ref["winkler"]["std"])
        verdict = "worse without it" if (gap > tol or wgap > wtol) else ("better without it" if (gap < -tol or wgap < -wtol) else "no clear effect")
        abl_rows.append([r["model"].replace("ablation:", "").strip().capitalize(),
                         f"{r['nrmse_pct']['mean']:.3f} ± {r['nrmse_pct']['std']:.3f}", f"{gap:+.3f}",
                         f"{r['winkler']['mean']:.4f}", f"{wgap:+.4f}", f"{r['picp']['mean']:.3f}", verdict])
    abl_rows.insert(0, ["Reference (all parts, no EMA)", f"{ref['nrmse_pct']['mean']:.3f} ± {ref['nrmse_pct']['std']:.3f}", "–",
                        f"{ref['winkler']['mean']:.4f}", "–", f"{ref['picp']['mean']:.3f}", "–"])
    R.table(["Variant", "nRMSE %", "Δ nRMSE", "Winkler", "Δ Winkler", "PICP (all)", "Effect"], abl_rows,
            "Ablation studies on Yorkshire (three seeds each). Δ is the change relative to the reference; "
            "'worse without it' means removing the part made the result worse beyond about two standard deviations.",
            widths=[4.6, 2.5, 1.5, 1.7, 1.7, 1.5, 2.2], size=8, align=[L_, C_, C_, C_, C_, C_, C_], highlight_rows=(1,))
    R.fig(FIG / "fig_7_4_ablation.png", "Change in nRMSE and Winkler score when a part is removed (grey = within noise; red = clearly worse; whiskers = ±2 standard deviations).", width_cm=15.0)
    R.body("**Only the dense expert loss showed a clear contribution:** without it, nRMSE rose from "
           f"{ref['nrmse_pct']['mean']:.2f}% to {R_ABL_BY('no dense expert loss')['nrmse_pct']['mean']:.2f}%, the "
           "Winkler score rose by about 8%, and the coverage fell (all-points PICP from 0.820 to 0.776). All other parts (the gate, the temperature "
           "control, the σ-dependent interval width, using only one of the two uncertainty signals, using one expert "
           "instead of four, and the extra training weight on each expert's own regime) made no measurable "
           "difference on this data. We therefore **do not claim** that routing or regime specialisation improves "
           "accuracy. The gate is not uniform in the final model (Figure 7.8): its mean entropy is "
           f"{d.test['routing']['mean_entropy']:.2f} against {d.test['routing']['max_entropy']:.2f} for uniform "
           "routing, and the Dawn-ramp expert has the largest share. But it does not follow the weather type; for "
           f"clear-midday cases the Clear-peak expert receives only {d.gate_mat[1][1]:.2f} of the weight while the "
           f"Dawn-ramp expert receives {d.gate_mat[1][0]:.2f}. The experts behave more like a small ensemble than like "
           "regime specialists.")
    R.body("We also tested other changes (Table 7.6). None of them improved on EMA alone: a larger weight decay, a "
           "smaller learning rate, an extra squared-error term on the median, and feeding the regime and volatility "
           "metadata into the shortcut all lay within the run-to-run variation of the reference, and combining "
           "them was no better than EMA alone.")
    R.table(["Change", "Runs", "nRMSE %", "Winkler"],
            [[n, str(k), f"{m:.3f} ± {s:.3f}", f"{wm:.4f}"] for n, k, m, s, wm, ws in d.opts],
            "Other training options tested (Yorkshire, three seeds each).",
            widths=[7.4, 1.4, 3.6, 2.6], size=9.5, align=[L_, C_, C_, C_], highlight_rows=(1,))

    R.h2("7.5", "Calibration and Coverage")
    R.body(f"About {d.day_share * 100:.0f}% of the forecast points lie in daylight. At night the true output and the "
           f"interval are both zero, so almost every night point counts as 'covered' ({d.picp_night * 100:.1f}%). "
           f"The all-points PICP of the raw model was therefore {d.picp_all:.3f}, apparently on target, while the "
           f"daytime coverage was only {d.picp_day:.3f}. Daytime coverage is the meaningful figure. "
           "Per-regime conformal calibration fitted on daytime validation points (Section 5.1.7) brought it to the "
           "within about one percentage point of the target on both measured regions (Table 7.7, Figures 7.5 and 7.6).")
    R.table(["Site", "Daytime coverage raw", "Daytime coverage calibrated", "Winkler raw", "Winkler calibrated", "PINAW raw", "PINAW calibrated"],
            [[n, f"{c['picp_daytime_before']:.3f}", f"{c['picp_daytime_after']:.3f}", f"{c['winkler_before']:.4f}",
              f"{c['winkler_after']:.4f}", f"{c['pinaw_before']:.4f}", f"{c['pinaw_after']:.4f}"]
             for n, c in (("Yorkshire", cy), ("South West", cs), ("Vellore (simulated)", cv))],
            "Effect of conformal calibration on one trained model (seed 42) per site; target coverage 0.80.",
            widths=[3.4, 2.1, 2.3, 1.8, 2.1, 1.8, 2.1], size=8.5, align=[L_, C_, C_, C_, C_, C_, C_])
    R.fig(FIG / "fig_7_5_daytime_coverage.png", "Share of daytime real values inside the P10–P90 range before and after calibration.", width_cm=11.6)
    R.fig(FIG / "fig_7_6_reliability_daytime.png", "Reliability of the P10, P50 and P90 quantiles on daytime points of the Yorkshire test set.", width_cm=8.6)
    R.body("Calibration widens the ranges (the Yorkshire PINAW rises from "
           f"{cy['pinaw_before']:.4f} to {cy['pinaw_after']:.4f}) but lowers the Winkler score slightly, because the "
           "reduction in missed observations outweighs the extra width. Two cautions apply. The volatile regime "
           "has only 44 validation windows, so its adjustment is noisy; and the validation and test periods fall in "
           "different seasons, so the coverage on new data may differ a little from the target. On the simulated "
           f"Vellore site calibration only reached {cv['picp_daytime_after']:.2f}, so we describe those ranges "
           "as still somewhat too narrow.")
    R.body("Table 7.8 gives accuracy and daytime coverage by weather situation for the main Yorkshire model, with "
           "LightGBM (seed 42) for comparison.")
    reg_rows = []
    cmp_ = d.conf["per_regime_comparison"]
    for r in d.regimes:
        a = d.test["per_regime"][r]
        lgm = d.lgbm1["per_regime"][r]
        reg_rows.append([d.REGIME_PLAIN[r], str(a["n"]), f"{a['nrmse_pct']:.2f}", f"{lgm['nrmse_pct']:.2f}",
                         f"{cmp_[r]['picp_daytime_before']:.2f}", f"{cmp_[r]['picp_daytime_after']:.2f}"])
    R.table(["Weather situation", "Windows", "nRMSE % ATERNET", "nRMSE % LightGBM", "Daytime coverage raw", "Daytime coverage calibrated"],
            reg_rows, "Results by weather situation (Yorkshire, seed 42, test set).",
            widths=[4.6, 1.8, 2.3, 2.3, 2.3, 2.3], size=9, align=[L_, C_, C_, C_, C_, C_])
    R.body("ATERNET and LightGBM are close in every situation; the largest errors occur in the volatile and ramp "
           "situations, as expected. The 'overcast' group includes night-time windows, which is why it is the "
           "largest.")

    R.h2("7.6", "Behaviour of the Forecasts and of the Uncertainty")
    R.body("Figure 7.7 shows one example for each weather situation. The best guess follows the real output and "
           "the shaded range widens where the output is harder to predict, as in the patchy-cloud case.")
    R.fig(FIG / "fig_7_7_forecast_examples.png", "Example forecasts for each weather situation (Yorkshire test set).", width_cm=15.0)
    R.body("Figure 7.8 shows how the gate distributes its trust among the experts in each real weather situation. "
           "A specialised pool would show a strong diagonal (each situation relying on its own expert); the "
           "figure shows that it does not, in agreement with the ablation results of Section 7.4.")
    R.fig(FIG / "fig_7_8_gate_by_regime.png", "Average weight given to each expert (columns) in each real weather situation (rows) on the test set.", width_cm=10.8)
    rt = d.test["routing"]
    R.body("The model's unsureness is informative: the combined uncertainty σ_total has a correlation of "
           f"{rt['corr_sigma_total_vs_abs_error']:.2f} with the real error of the forecast, and the width of the "
           f"forecast range correlates {rt['corr_interval_width_vs_abs_error']:.2f} with it (Figure 7.9). In other words, "
           "wider ranges are really drawn where the errors are larger.")
    R.fig(FIG / "fig_7_9_uncertainty_vs_error.png", "Average forecast error against the model's unsureness score and against the range width.", width_cm=15.0)
    R.body(f"Figure 7.10 shows the error by forecast horizon. In the first hour ATERNET and LightGBM are equal "
           f"({d.hz_at[1] * 100:.2f}% and {d.hz_lg[1] * 100:.2f}% at one hour). ATERNET is more accurate from about "
           f"one and a half to five hours ahead (for example {d.hz_at[3] * 100:.2f}% against {d.hz_lg[3] * 100:.2f}% "
           f"at two hours), the two are similar around five to six hours, and at the longest horizons ATERNET is "
           f"slightly worse ({d.hz_at[-1] * 100:.2f}% against {d.hz_lg[-1] * 100:.2f}% at eight hours). The "
           "long-horizon error is dominated by the quality of the weather forecast.")
    R.fig(FIG / "fig_7_10_error_by_horizon.png", "RMSE by forecast horizon for ATERNET and LightGBM (seed 42, Yorkshire).", width_cm=12.4)
    import json
    hist = js(M / "train_history.json")
    best = min(range(len(hist)), key=lambda i: hist[i]["val_pinball_total"])
    R.body(f"Training (Figure 7.11) stopped automatically after {len(hist)} epochs with the best validation loss at "
           f"epoch {hist[best]['epoch']}; an epoch took about 12 seconds on the CPU.")
    R.fig(FIG / "fig_7_11_training_history.png", "Training and validation loss of the main model.", width_cm=12.4)

    R.h2("7.7", "Out-of-Sample Backtest")
    b = d.back
    R.body("To test the complete live pipeline, forecasts were issued six times per day for 1–16 October 2026, after "
           "the end of the training period, using only data available at each issue time, and scored against what "
           "PV_Live later reported (Table 7.9, Figure 7.12).")
    R.table(["Quantity", "Value"],
            [["Forecasts issued / half-hours scored", f"{b['n_issue_times']} / {b['n_steps']:,}"],
             ["nMAE / nRMSE (% of capacity)", f"{b['nmae_pct']:.2f} / {b['nrmse_pct']:.2f}"],
             ["R²", f"{b['r2']:.3f}"],
             ["Daylight MAE", f"{b['daylight_mae_mw']:.1f} MW (mean output {b['daylight_mean_actual_mw']:.0f} MW)"],
             ["Day-ahead persistence daylight MAE", f"{b['daylight_persistence_mae_mw']:.1f} MW"],
             ["Skill against persistence", f"{b['rmse_skill_vs_persistence']:.3f}"],
             ["Coverage of the P10–P90 range (all points)", f"{b['picp']:.3f}"]],
            "Backtest of the live pipeline, Yorkshire, 1–16 October 2026.", widths=[8.6, 7.0], size=9.5, align=[L_, C_])
    R.fig(FIG / "fig_7_12_backtest.png", "Backtest: P10–P90 range and best guess two hours ahead against the measured output.", width_cm=15.2)
    R.body(f"The daylight mean absolute error is {b['daylight_mae_mw']:.0f} MW, about "
           f"{b['daylight_persistence_mae_mw'] / b['daylight_mae_mw']:.1f} times better than persistence "
           f"({b['daylight_persistence_mae_mw']:.0f} MW). The range covered {b['picp'] * 100:.0f}% of the points, "
           "somewhat below 0.80, which is consistent with the slightly narrow raw ranges found in Section 7.5. Two "
           "caveats apply: the historical weather archive is slightly better than a single live forecast, so live "
           "errors may be a little larger, and PV_Live is a regional aggregate that is smoother than one plant.")

    R.h2("7.8", "India Demonstration with Simulated Data")
    vit = d.vit_test["headline"]
    vA, vL = d.VIT["ATERNET (final configuration)"], d.VIT["LightGBM quantile"]
    R.body("No open measured PV record exists for the VIT Vellore area, and we could not obtain data from the "
           "campus, so a demonstration site was built (Section 5.1.11). **All numbers in this section describe "
           "how well the model learns a simulated plant from real weather forecasts. They are not measured "
           "results and must not be read as real-world accuracy for India.** Training used January 2024 to "
           "September 2026 and four weather models (the fifth does not cover Vellore before 2025).")
    R.table(["Model", "nRMSE %", "nMAE %", "R²", "Winkler", "PINAW"],
            [["ATERNET (seed 42)", f"{vA['nrmse_pct']['mean']:.2f}", f"{vA['nmae_pct']['mean']:.2f}", f"{vA['r2']['mean']:.4f}", f"{vA['winkler']['mean']:.4f}", f"{vA['pinaw']['mean']:.4f}"],
             ["LightGBM (seed 42)", f"{vL['nrmse_pct']['mean']:.2f}", f"{vL['nmae_pct']['mean']:.2f}", f"{vL['r2']['mean']:.4f}", f"{vL['winkler']['mean']:.4f}", f"{vL['pinaw']['mean']:.4f}"]],
            "Simulated Vellore plant (1 MWp), one training run each; test windows: 1,805.",
            widths=[4.0, 2.2, 2.2, 2.4, 2.4, 2.4], size=9.5, align=[L_, C_, C_, C_, C_, C_])
    R.fig(FIG / "fig_7_13_vellore_examples.png", "Simulated Vellore site: forecast ranges and the naive 'same as now' guess against the simulated output.", width_cm=15.2)
    R.body(f"On this site ATERNET scored an nRMSE of {vA['nrmse_pct']['mean']:.2f}% against "
           f"{vL['nrmse_pct']['mean']:.2f}% for LightGBM, and a lower Winkler score. Each model was trained only "
           "once, so this is the outcome of one run and is not a statistically tested win. Daytime coverage was "
           f"{cv['picp_daytime_before']:.2f} before and {cv['picp_daytime_after']:.2f} after calibration, i.e. "
           "short of the 0.80 target.")

    R.h2("7.9", "Additional Experiment: Denser Training Windows")
    R.body("Does more training data help? We built a training set with twice as many (overlapping) windows by halving "
           "the stride, keeping the validation and test sets identical so the results stay comparable, and re-ran "
           "ATERNET and LightGBM with seeds 42 and 43 (Table 7.11).")
    R.table(["Model", "Stride 4 (seeds 42, 43)", "Stride 2 (seeds 42, 43)"],
            [["ATERNET nRMSE %", ", ".join(f"{v:.2f}" for v in d.s4_at), ", ".join(f"{v:.2f}" for v in d.s2_at)],
             ["LightGBM nRMSE %", ", ".join(f"{v:.2f}" for v in d.s4_lg), ", ".join(f"{v:.2f}" for v in d.s2_lg)]],
            "Effect of doubling the number of training windows (Yorkshire).",
            widths=[4.6, 5.5, 5.5], size=9.5, align=[L_, C_, C_])
    R.body("LightGBM improved with the extra windows, while ATERNET did not (it became slightly worse). With more "
           "windows LightGBM therefore pulls ahead on accuracy. The training schedule was not re-tuned for the "
           "larger set, so this does not prove that ATERNET cannot use more data, but it shows that the accuracy tie "
           "with LightGBM holds at the original data density and should not be generalised.")

    R.h2("7.10", "Discussion")
    R.bullets([
        "**What the evidence supports.** ATERNET is as accurate as the strongest baseline on Yorkshire, more accurate "
        "on South West England, stable across training runs, and its forecast ranges have a better Winkler score than "
        "every stable baseline. After calibration its daytime coverage is within about one percentage point of the 0.80 target on both measured regions.",
        "**What it does not support.** It does not show that adaptive routing or regime-specialised experts improve "
        "accuracy; the ablations show they do not, and only the dense expert loss was shown to matter. It does not "
        "show that ATERNET beats a well-trained LSTM, or that it would stay ahead of LightGBM with more training data.",
        "**Threats to validity.** (i) Only two regions in England. (ii) Baselines were run with defaults and not tuned; "
        "tuning could narrow the gaps. (iii) Three seeds give only rough estimates of variation. (iv) Several design "
        "choices (such as EMA) were selected using results on the Yorkshire test period, which creates a mild "
        "selection bias there; the second region is the unbiased check. (v) The regime labels are rule-based and "
        "night-time windows default to 'overcast'. (vi) The India numbers are simulated.",
        "**Lessons.** Features that sound principled, such as an adaptive gate, must be tested against a plain "
        "average over several seeds; coverage must be measured where it matters (daylight); and single-run "
        "comparisons can be wrong by large amounts.",
    ])

    R.h2("7.11", "Cost Analysis")
    R.body("The project required no purchase of software, data or cloud services (Table 7.12). The costs are the time "
           "of the developer and the electricity used by a personal laptop; no monetary figure is claimed because "
           "none was recorded.")
    R.table(["Item", "Cost / resource", "Note"],
            [["Software (Python, PyTorch, LightGBM, Streamlit, etc.)", "Free, open source", "Licences permit research use"],
             ["PV data (Sheffield Solar PV_Live)", "Free", "Public API, no key"],
             ["Weather data (Open-Meteo, ERA5)", "Free", "Public APIs; non-commercial use"],
             ["Computing", "One laptop (i5-12450H, 7.7 GB)", "No GPU or cloud needed"],
             ["Compute time", "About 10 minutes per training run", "More than 60 runs in total, of the order of 20 hours"],
             ["Storage", "A few GB", "Datasets, checkpoints, cached downloads"],
             ["Deployment (if adopted)", "One small server or an existing PC", "Model has about 968,000 parameters"]],
            "Cost and resource summary.", widths=[6.4, 4.4, 4.8], size=9.5, align=[L_, L_, L_])
    R.body("For an operator, the benefit would come from reduced reserve requirements and better trading decisions; "
           "estimating it would need operator-specific data that we do not have, so this report makes no such claim.")


def R_ABL():
    return [r for r in d_rows() if r["kind"] == "ablation"]


def d_rows():
    from report_data import SUMMARY
    return SUMMARY["sites"]["yorkshire"]["rows"]


def R_ABL_BY(name):
    for r in R_ABL():
        if name in r["model"]:
            return r
    raise KeyError(name)


def chapter8(R):
    y, sw = d.Y[A], d.SW[A]
    ylg = d.Y["LightGBM quantile"]
    slg = d.SW["LightGBM quantile"]
    cy, cs = d.cal["yorkshire"], d.cal["south_west"]
    R.chapter_heading(8, "Conclusion")
    R.body("This project designed, built and evaluated ATERNET, a probabilistic solar power forecaster that "
           "combines a patch-based Transformer backbone, four regime experts, an uncertainty-aware router and a "
           "probabilistic decoder, together with a complete, tested pipeline from public data to an explanatory "
           "dashboard.")
    R.body("**Main findings.** (1) On measured regional data from Yorkshire, ATERNET reaches an nRMSE of "
           f"{fmt(y, 'nrmse_pct', 2)}%, equal to LightGBM ({fmt(ylg, 'nrmse_pct', 2)}%) and better than PatchTST, with a "
           "better Winkler score and very stable training. (2) On South West England, without re-tuning, it is "
           f"ahead of both baselines ({fmt(sw, 'nrmse_pct', 2)}% against {fmt(slg, 'nrmse_pct', 2)}% for LightGBM). "
           "(3) The original learned gate harmed accuracy and stability; a residual gate, a last-step linear "
           "shortcut and weight averaging were what produced the final accuracy. (4) Interval coverage measured over "
           "all hours was misleading; with daytime-only calibration the daytime coverage reaches "
           f"{cy['picp_daytime_after']:.2f} and {cs['picp_daytime_after']:.2f} against a target of 0.80. (5) Of the "
           "architectural ideas, only the dense expert loss showed a measurable benefit; adaptive routing and regime "
           "specialisation did not improve accuracy.")
    R.body("**Contributions.** The project contributes a working probabilistic forecaster with built-in, tested "
           "guarantees; a reproducible multi-seed evaluation protocol with plain-language reporting; a worked example of "
           "diagnosing and correcting a harmful gate and a coverage statistic that overstated reliability; a "
           "clearly labelled simulated-site mechanism for places without data; and an understandable dashboard.")
    R.body("**Limitations.** The evidence comes from two English regions; baselines were not tuned; seed counts are small; "
           "the Indian demonstration is simulated; and LightGBM benefits more than ATERNET from additional training "
           "windows.")
    R.body("**Future work.** (i) Obtain measured PV data for an Indian site, for example from a campus plant, and "
           "retrain or fine-tune on it. (ii) Extend to more regions and seasons. (iii) Tune the baselines equally and "
           "test ensembles of several ATERNET runs. (iv) Make the experts specialise genuinely, for example "
           "with a diversity penalty or regime-specific data, and test again whether routing then helps. "
           "(v) Extend the horizon, add satellite-based inputs and deploy the forecaster as a service.")


def _src(path, names):
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            seg = ast.get_source_segment(src, node)
            # methods keep their original indentation; re-indent the first line
            first_indent = " " * node.col_offset
            out.append(first_indent + seg)
    return "\n\n".join(out)


def appendix(R):
    R.plain_heading("APPENDIX A – Sample Code", toc_title="APPENDIX A – SAMPLE CODE", size=14)
    R.body("Selected excerpts of the source code are given below to document the key mechanisms. The complete code "
           "is in the project repository.", after=8)
    R.body("**A.1 Residual gate with uncertainty-dependent temperature** (src/phase4/gating_network.py).", align=L_, keep_next=True)
    R.code(_src("src/phase4/gating_network.py", {"GatingNetwork"}))
    R.body("**A.2 Weight averaging (EMA) in the trainer** (src/training/trainer.py).", align=L_, before=10, keep_next=True)
    R.code(_src("src/training/trainer.py", {"_ema_update", "_ema_weights"}))
    R.body("**A.3 Daytime mask and conformity scores for calibration** (src/evaluation/conformal.py).", align=L_, before=10, keep_next=True)
    R.code(_src("src/evaluation/conformal.py", {"daytime_mask", "_conformity_scores"}))
    R.body("**A.4 Forward pass of ATERNET with the last-step shortcut** (src/models/aternet.py).", align=L_, before=10, keep_next=True)
    R.code(_src("src/models/aternet.py", {"forward"}))
    R.body("**A.5 Simulated PV model for sites without measurements** (src/phase1/simulated_pv.py).", align=L_, before=10, keep_next=True)
    R.code(_src("src/phase1/simulated_pv.py", {"pv_power_kw"}))
