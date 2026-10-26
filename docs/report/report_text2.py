"""Chapters 4 and 5: design, methodology and testing."""

from report_data import d, FIG
from report_lib import C_, J, L_, R_


def chapter4(R):
    R.chapter_heading(4, "Design Approach and Details")
    R.h2("4.1", "System Architecture")
    R.body("The system is organised in four layers (Figure 4.1). The **data layer** downloads measured PV output and "
           "forecast weather, cleans and aligns them, derives features and cuts them into windows. The **model and "
           "experiment layer** contains the ATERNET network, its trainer, the baseline models, the evaluation code and "
           "the calibration step. The **application layer** exposes the results through a command-line forecaster "
           "and an interactive dashboard, and a small results store makes the numbers available to the dashboard "
           "without re-running experiments.")
    R.fig(FIG / "fig_4_1_system_architecture.png", "System architecture of the ATERNET forecasting system.",
          width_cm=13.6, max_h_cm=17.5)
    R.body("**The ATERNET network** (Figure 4.2) is built from five phases that are trained together. "
           "Three input tensors enter: the last 24 hours of measured and derived signals (X_seq, 48 steps × 10 "
           "features), the weather forecasts for the past 24 hours and the next 8 hours (X_nwp, 64 × 4) and "
           "the forecast-spread features that describe how much the NWP models disagree (X_sigma, 48 × 6).")
    R.fig(FIG / "fig_4_2_model_architecture.png",
          "Architecture of the ATERNET model: backbone, expert pool, adaptive router and probabilistic decoder.",
          width_cm=12.4, max_h_cm=17.5)
    R.bullets([
        "**Phase 2, temporal backbone.** The measured sequence is cut into overlapping patches (8 steps, stride 4), "
        "embedded, given a learnable position and passed through a three-layer Transformer encoder with eight "
        "attention heads. A cross-attention block lets the PV patches look at the forecast-weather patches, and a "
        "small network encodes the forecast-spread features. The backbone produces three embeddings: h_expert (128 "
        "values) for the experts, h_nwp (32) for the decoder and h_sigma (32) for the router.",
        "**Phase 3, expert pool.** Four expert networks, one for each weather regime (morning ramp-up, clear "
        "midday, overcast, patchy clouds), each predict the 16 future values from h_expert. The experts differ in "
        "capacity, loss and physical constraint (Table 5.3). Their disagreement is measured as σ_expert, the "
        "standard deviation across experts, computed on detached predictions so that it cannot be gamed by the experts.",
        "**Phase 4, adaptive router.** Two uncertainty signals are combined: σ_NWP, the spread between the NWP "
        "models, and σ_expert, the disagreement between experts. A learned mixing weight α depends on the "
        "embedding, so the model decides when to trust which signal. The combined σ_total sets the temperature of a "
        "softmax gate that weights the experts, and also controls the width of the final interval.",
        "**Phase 5, probabilistic decoder.** A bounded correction is added to the routed forecast to give the "
        "median (P50). Separate lower and upper half-widths that increase with σ_total give P10 and P90, so the "
        "quantiles can never cross.",
    ])
    R.body("The uncertainty mixing and the temperature are given by")
    R.equation("σ_total = α · σ̃_NWP + (1 − α) · σ̃_exp ,   α = sigmoid(wᵀ h_expert + b)")
    R.equation("T = exp( softplus(β) · σ_total ) ,   weights = softmax( δ · z / T )")
    R.body("where σ̃ denotes a normalised uncertainty, z the gate's raw scores for the four experts, β a learned "
           "sensitivity and δ a learned scale (explained below). A large σ_total therefore raises the temperature T "
           "and spreads the votes across experts, as a hedge. The routed forecast and the interval are")
    R.equation("ŷ_routed = Σₖ weightsₖ · ŷₖ + S(x_last, x_nwp,future)")
    R.equation("lower_h = softplus(a_h + softplus(γ_h) · σ_total)   (upper_h analogous)")
    R.equation("P10 = clip(P50 − lower) ,   P90 = clip(P50 + upper)")
    R.body("The term S in the first equation is the **last-step shortcut**, a single linear layer from the latest "
           "observed step and the forecast weather of the next eight hours to the 16 outputs. Its weights start "
           "at zero, so at the start of training the model is identical to one without it.")
    R.body("**Three structural guarantees** are built into the design and checked by automated tests: (G1) P10 ≤ P50 ≤ "
           "P90 always holds, because the widths are positive by construction; (G2) the interval width is "
           "non-decreasing in σ_total; and (G3) the gate softens (spreads its votes) when σ_total grows.")
    R.body("**The two design changes that came from our experiments** are explained in Chapter 7. First, the "
           "original gate, which learned its scores directly, was replaced by a **residual gate** in which the "
           "scores are multiplied by a learnable scalar δ initialised to zero, so that routing starts as an exact "
           "uniform average and departs from it only if that lowers the validation loss (the same principle as "
           "ReZero [@bachlechner2021]). Second, the optional shortcut and an exponential moving average of the "
           "weights (Section 5.1.6) were added.")

    R.h2("4.2", "Design")
    R.h3("4.2.1", "Data Flow Diagram")
    R.body("The context diagram (Figure 4.3) shows the system as a single process with its external entities: the two "
           "data services supply measured PV and weather; researchers supply configuration and training requests; "
           "reviewers and grid analysts receive results, comparisons and forecasts with ranges.")
    R.fig(FIG / "fig_4_3_dfd_level0.png", "Data flow diagram, level 0 (context diagram).", width_cm=13.5)
    R.body("The level-1 diagram (Figure 4.4) decomposes the system into six processes and four data stores. "
           "Process 1.0 acquires raw data into D1; process 2.0 cleans, engineers features and builds windows into D2; "
           "process 3.0 trains the model and saves the best weights into D3; process 4.0 evaluates and calibrates "
           "and writes metrics to D4; process 5.0 produces live or backtest forecasts from the stored model; "
           "process 6.0 presents everything in the dashboard.")
    R.fig(FIG / "fig_4_4_dfd_level1.png", "Data flow diagram, level 1.", width_cm=13.6, max_h_cm=17.0)

    R.h3("4.2.2", "Use Case Diagram")
    R.body("Figure 4.5 shows the actors and use cases. The researcher or student prepares data, trains and evaluates "
           "models and runs studies; the reviewer or examiner explores results; the grid or energy analyst consumes "
           "forecasts. The two data services take part in the use cases that need external data.")
    R.fig(FIG / "fig_4_5_use_case.png", "Use case diagram of the ATERNET forecasting system.", width_cm=12.4,
          max_h_cm=15.5)
    R.table(
        ["Use case", "Actor(s)", "Summary"],
        [
            ["UC1 Download and prepare data", "Researcher; PV_Live, Open-Meteo", "Fetch PV and weather, clean, build windows and scalers"],
            ["UC2 Train the model", "Researcher", "Joint training with early stopping and weight averaging"],
            ["UC3 Evaluate against baselines", "Researcher, Reviewer", "Metrics for ATERNET, LightGBM, PatchTST, LSTM"],
            ["UC4 Calibrate forecast ranges", "Researcher", "Per-regime conformal adjustment on daytime validation points"],
            ["UC5 Run multi-seed / ablation study", "Researcher", "Repeat training over seeds, switch off model parts"],
            ["UC6 Generate live forecast", "Analyst; PV_Live, Open-Meteo", "Forecast the next 8 hours from the latest data"],
            ["UC7 Replay past forecasts", "Analyst", "Backtest against measured output (includes UC6)"],
            ["UC8 Explore results in dashboard", "Researcher, Reviewer, Analyst", "Interactive pages with plain-language explanations"],
        ],
        "Use cases and their actors.", widths=[5.2, 4.4, 6.0], size=9.5, align=[L_, L_, L_])

    R.h3("4.2.3", "Class Diagram")
    R.body("Figure 4.6 shows the main classes. ATERNET is composed of the backbone, the expert pool, the router and "
           "the decoder; the expert pool aggregates four concrete experts that inherit from an abstract ExpertBase "
           "(which defines the primary loss, the physical-prior auxiliary loss and the combined loss); the router "
           "is composed of the uncertainty-fusion module and the gating network. The Trainer uses the dataset to "
           "train ATERNET, and the Forecaster serves it for live and backtest forecasts.")
    R.fig(FIG / "fig_4_6_class_diagram.png", "Class diagram of the main software components.", width_cm=13.8,
          max_h_cm=18.0)

    R.h3("4.2.4", "Sequence Diagram")
    R.body("Figure 4.7 shows the sequence for a live forecast: the Forecaster obtains the latest PV and weather from "
           "the two services, aligns them to the 30-minute grid, builds and scales one window, calls the model and "
           "returns a table of P10, P50 and P90 in megawatts.")
    R.fig(FIG / "fig_4_7_sequence_live.png", "Sequence diagram of a live forecast request.", width_cm=14.2, max_h_cm=16)
    R.body("Figure 4.8 shows the sequence for training and evaluation. For every epoch the Trainer runs forward and "
           "backward passes, updates the optimiser and the averaged weights, validates with the averaged weights and "
           "keeps the best checkpoint; evaluation and calibration are then run on the saved model.")
    R.fig(FIG / "fig_4_8_sequence_training.png", "Sequence diagram of training, evaluation and calibration.",
          width_cm=14.2, max_h_cm=16)


def chapter5(R):
    R.chapter_heading(5, "Methodology and Testing")
    R.h2("5.1", "Module Description")
    R.h3("5.1.1", "Data Acquisition and Preparation (Phase 1)")
    R.body("**Data sources.** Measured regional PV generation comes from Sheffield Solar PV_Live [@pvlive], which "
           "reports half-hourly generation and installed capacity for each distribution-network region of Great Britain. "
           "Weather comes from the Open-Meteo forecast archive [@openmeteo], which stores the forecasts that "
           "five independent NWP systems actually issued. We treat each system as one ensemble member, so the spread "
           "between them is a genuine measure of forecast uncertainty. Weather is sampled at three points across a "
           "region and averaged. The same models are available from the live forecast API, so the inputs seen "
           "in training and in live use have the same form.")
    R.table(
        ["Item", "Yorkshire", "South West England"],
        [
            ["PV_Live region", "23 (NPG Yorkshire)", "22 (WPD South West)"],
            ["Installed capacity (end of period)", "1,165 MWp", "2,693 MWp"],
            ["Period", "1 Jan 2023 – 30 Sep 2026", "1 Jan 2023 – 30 Sep 2026"],
            ["Half-hours / windows", "65,713 / 16,413", "65,713 / 16,413"],
            ["Train / validation / test windows", "11,489 / 2,462 / 2,462", "11,489 / 2,462 / 2,462"],
            ["Train period", "1 Jan 2023 – 15 Aug 2025", "same grid"],
            ["Validation period", "15 Aug 2025 – 8 Mar 2026", "same grid"],
            ["Test period", "8 Mar – 29 Sep 2026", "same grid"],
            ["Weather members", "GFS, ICON, UKMO, Météo-France, GEM", "same"],
        ],
        "Datasets used for the two measured regions.", widths=[5.6, 5.0, 5.0], size=9.5, align=[L_, C_, C_])
    R.body("**Cleaning.** The generation values are converted to power and normalised by the installed capacity "
           "(which grows over time), clipped to [0, 1], checked for gaps, and the solar elevation is computed for "
           "every half-hour with the NOAA formulation, including longitude and the equation of time.")
    R.body("**Features** (Table 5.2) include the clear-sky index (measured or forecast irradiance divided by the "
           "clear-sky value), time-of-day and day-of-year encodings, the ensemble mean of each weather variable, and "
           "the spread between the weather models.")
    R.table(
        ["Group", "Shape", "Contents"],
        [
            ["X_seq", "48 × 10", "pv_norm, ghi, cloud cover, temperature, wind speed, clear-sky index, hour sin/cos, day-of-year sin/cos"],
            ["X_sigma", "48 × 6", "σ_NWP for irradiance, cloud, temperature and wind; a placeholder for σ_expert; normalised ensemble spread"],
            ["X_nwp", "64 × 4", "Ensemble-mean forecast of irradiance, cloud, temperature, wind over the past 24 h and the next 8 h"],
            ["y", "16", "Capacity-normalised PV output for the next 8 hours (30-minute steps)"],
            ["meta", "9", "Regime id and one-hot, ramp-rate and rolling volatility statistics (used for stratification and evaluation)"],
        ],
        "Input and target tensors built in Phase 1.", widths=[2.2, 2.2, 11.2], size=9.5, align=[C_, C_, L_])
    R.body("**Weather regimes.** For stratified evaluation and for the experts, each window is tagged by simple rules "
           "at the issue time: dawn ramp (solar elevation between 2° and 20° with a clear ramp rate), clear peak "
           "(clear-sky index above 0.75 and low one-hour volatility), overcast (clear-sky index at most 0.4 and low "
           "volatility) and volatile (one-hour standard deviation of at least 0.05). Windows issued at night "
           "default to the overcast tag, which is why that regime is the largest in the test set (Table 7.8).")
    R.fig(FIG / "fig_5_1_data_pipeline_flow.png", "Flow of data acquisition and preparation.", width_cm=10.6, max_h_cm=17.5)
    R.body("**Windows and splits.** Each sample is 48 half-hours of history and 16 half-hours of future, taken every "
           "4 steps (two hours). The data are split chronologically, 70% for training, 15% for validation and 15% "
           "for testing, with no shuffling, and the min-max scalers are fitted on the training part only so no "
           "information from the future reaches the model. Figure 5.2 shows one week of measured output and the "
           "corresponding forecast irradiance with the spread between the five weather models.")
    R.fig(FIG / "fig_5_3_data_week.png",
          "One week (8–14 June 2026) of measured PV output in Yorkshire and the ensemble forecast of irradiance with the spread between the models.",
          width_cm=14.6)

    R.h3("5.1.2", "Temporal Backbone")
    R.body("The backbone follows the PatchTST design [@nie2023]. A patch length of 8 steps with a stride of 4 turns the "
           "48-step history into 11 patches that are projected to 128 dimensions and given learnable positions. A "
           "three-layer, pre-norm Transformer encoder with eight heads and a feed-forward size of 256 processes the "
           "patches. The weather forecast is cut into patches in the same way and attended to by the PV patches. "
           "The forecast-spread features, which have a physical meaning of their own, are encoded by a separate "
           "small network and are deliberately kept out of the Transformer.")
    R.h3("5.1.3", "Expert Pool")
    R.body("The four experts are sized to the difficulty of their weather regime (Table 5.3). All four are run on every "
           "sample, and every expert also receives its own training loss on every sample (the dense expert loss), "
           "with the samples of its own regime weighted three times higher. This keeps all experts learning "
           "even when the gate gives them little weight.")
    R.table(
        ["Expert", "Hidden sizes", "Loss", "Built-in constraint", "Parameters"],
        [
            ["Dawn ramp", "256→256→128", "MSE + slope hinge", "ramp as residual + cumulative sum", "134,929"],
            ["Clear peak", "128→64", "MSE", "none", "26,064"],
            ["Overcast", "128→128", "Huber (δ = 0.05)", "output capped at 0.80", "35,344"],
            ["Volatile", "512→256→128", "MSE", "none; widest, most dropout", "233,872"],
            ["Pool total", "", "", "", "430,209"],
        ],
        "The four experts.", widths=[2.5, 3.0, 3.3, 4.7, 2.1], size=9.5, align=[L_, C_, C_, L_, C_])
    R.h3("5.1.4", "Adaptive Router")
    R.body("σ_NWP is the mean of the five forecast-spread features at the issue time and is not learned. σ_expert is "
           "passed through a single positive scale so its ranking is preserved. Both are variance-stabilised "
           "(log1p), batch-normalised and mixed with the learned weight α (Equation 4.1). The gate is a small network "
           "over the embeddings and uncertainty signals that returns one score per expert; the scores are divided by "
           "the temperature and, in the final model, scaled by the residual factor δ (Equation 4.2).")
    R.h3("5.1.5", "Probabilistic Decoder")
    R.body("The prediction head adds a bounded correction (at most ±0.1 after a tanh) to the routed forecast to give "
           "the median. The interval-width head outputs separate lower and upper half-widths for every step through "
           "a softplus, with a learned sensitivity to σ_total. P10 and P90 are the median minus the lower and plus "
           "the upper width, clipped to the physical range [0, 1].")
    R.h3("5.1.6", "Training Procedure")
    R.body("All phases are trained jointly with one optimiser, because the usefulness of the router depends on the "
           "experts and vice versa. The loss is the weighted sum")
    R.equation("L = L_pinball + 0.05 · L_sharp + w(t) · Σₖ L_expert,k + 0.01 · L_balance")
    R.body("The pinball loss at a fixed level τ is a proper scoring rule [@koenker1978,gneiting2007]; L_sharp is the "
           "mean interval half-width, which stops the model from reaching good coverage simply by drawing very wide "
           "intervals; L_expert is the dense loss of each expert with its physical-prior term (weight 0.1); and "
           "L_balance is a small penalty against the gate collapsing onto one expert. The dense-loss weight w(t) is "
           "1.0 for the first five epochs, decreases linearly to 0.3 over the next five and then stays at 0.3, so "
           "that the routed objective dominates once the experts are competent.")
    R.table(
        ["Setting", "Value"],
        [
            ["Optimiser", "AdamW [@loshchilov2019], learning rate 1e-3, weight decay 1e-4"],
            ["Learning-rate schedule", "Cosine annealing to 1% of the initial rate"],
            ["Batch size / epochs", "64 / at most 50"],
            ["Gradient clipping", "Global norm 1.0"],
            ["Early stopping", "Patience 8 epochs on the validation pinball loss"],
            ["Model selection", "Best validation pinball loss (not the composite loss, whose weights change)"],
            ["Weight averaging", "EMA with decay 0.999 and warm-up min(0.999, (1+t)/(10+t)) [@polyak1992]; validation and checkpoints use the averaged weights"],
            ["Window / horizon / stride", "48 / 16 / 4 half-hours"],
            ["Backbone", "Patch 8, stride 4, d_model 128, 8 heads, 3 layers, feed-forward 256, dropout 0.1"],
            ["Random seed", "42 for the main model; 42, 43, 44 for multi-seed studies"],
        ],
        "Training configuration of the final model.", widths=[4.4, 11.2], size=9.5, align=[L_, L_])
    R.fig(FIG / "fig_5_2_training_flow.png", "Flow of training, evaluation and calibration.", width_cm=10.2, max_h_cm=17.5)
    R.body("**Exponential moving average.** After every optimiser step a smoothed copy of the weights is updated as "
           "θ_ema ← d · θ_ema + (1 − d) · θ, with a warm-up that starts the decay low so the average does not remain "
           "close to the random initial weights. Validation, checkpointing and prediction use the smoothed weights, "
           "which proved to be one of the most effective changes in this work (Section 7.4).")
    R.h3("5.1.7", "Interval Calibration")
    R.body("The raw P10–P90 range of a model is not guaranteed to hold 80% of the observations. We use conformalized "
           "quantile regression [@romano2019]. For each weather regime and each daytime point of the validation "
           "set we compute the conformity score")
    R.equation("E = max( P10 − y , y − P90 )")
    R.body("and take its finite-sample quantile at level ⌈(n + 1)(1 − α)⌉ / n with α = 0.20. The resulting value is "
           "subtracted from P10 and added to P90 on daytime points of the test set, and the interval is kept "
           "non-crossing and inside [0, 1]. Two details matter. First, scores are computed for each forecast step, "
           "not averaged over the 16 steps of a window, because an average of signed violations is close to zero "
           "or negative and produces no adjustment. Second, only daytime points are used and adjusted, where "
           "daytime means the forecast irradiance input exceeds a small threshold; at night the truth and the "
           "interval are both zero and are trivially 'covered', which would otherwise inflate every coverage "
           "figure. Both were defects of our first implementation (Table 5.8).")
    R.h3("5.1.8", "Baseline Models")
    R.table(
        ["Model", "Description"],
        [
            ["Persistence", "The next hours equal the last observed value; used for skill scores"],
            ["LightGBM [@ke2017]", "One gradient-boosted model per quantile and per horizon step (48 models), 63 leaves, learning rate 0.05, up to 500 rounds with early stopping; 93 input features: latest step, forecast weather for the horizon, forecast-spread and metadata"],
            ["PatchTST [@nie2023]", "Transformer encoder on patches with three quantile outputs; no experts or routing"],
            ["LSTM [@hochreiter1997]", "Two layers of 128 units, dropout 0.1, concatenated inputs, three quantile heads, early stopping with patience 10"],
        ],
        "Baseline models used for comparison.", widths=[3.6, 12.0], size=9.5, align=[L_, L_])
    R.body("All baselines read exactly the same windows, splits and scalers as ATERNET, so they receive the same "
           "forecast weather, including the next eight hours. They were run with the default settings listed above and "
           "were not specially tuned, a limitation we discuss in Chapter 7.")
    R.h3("5.1.9", "Evaluation Metrics")
    R.table(
        ["Metric", "Meaning", "Better when"],
        [
            ["nRMSE (%)", "Root-mean-square error as a percentage of installed capacity", "lower"],
            ["nMAE (%)", "Mean absolute error as a percentage of capacity", "lower"],
            ["R²", "Share of the variance of the observations explained by the P50 forecast", "higher"],
            ["Skill vs persistence", "1 − RMSE / RMSE of the persistence forecast", "higher"],
            ["PICP", "Share of observations inside [P10, P90]; the target is 0.80", "close to 0.80"],
            ["PINAW", "Mean interval width as a fraction of capacity", "lower (at equal coverage)"],
            ["Winkler score", "Mean of width + (2/α) × (distance outside the interval), α = 0.2 [@winkler1972]", "lower"],
            ["Pinball loss", "Mean quantile loss at levels 0.1, 0.5, 0.9", "lower"],
        ],
        "Evaluation metrics.", widths=[3.2, 9.8, 2.6], size=9.5, align=[L_, L_, C_])
    R.body("PICP is reported in two ways: over all points, as is common in the literature, and over daytime points "
           "only. Winkler and the error metrics are computed over all points for every model in the same way, "
           "so they are comparable between models. Differences between models are called significant only when they "
           "exceed about twice the run-to-run standard deviation.")
    R.h3("5.1.10", "Inference and Dashboard")
    R.body("The Forecaster class builds a window from the latest data, scales it with the saved scalers and calls the "
           "model; the backtest command replays past issue times and scores them against what was measured. The "
           "dashboard (Chapter 6) is a Streamlit application that reads the saved predictions, the multi-seed "
           "summary and the calibration results, and re-computes the built-in safety checks live on the model.")
    R.h3("5.1.11", "Second Region and Simulated Site")
    R.body("A second measured region and a simulated site were added by configuration only. Environment variables "
           "choose the region and move the data, checkpoints and outputs to a separate workspace so the default "
           "experiments cannot be overwritten. For VIT Vellore, where no measured record exists, "
           "the PV output is simulated by a documented physical model (temperature derating, soiling, inverter "
           "clipping and about 1% noise) driven by ERA5 reanalysis weather [@hersbach2020], while the model's inputs "
           "remain the forecast models. Using a different weather source for the 'truth' and for the inputs "
           "ensures that the forecast-versus-actual error is real and that the task is not simply inverting "
           "a formula applied to the inputs.")

    R.h2("5.2", "Testing")
    R.h3("5.2.1", "Automated Tests")
    R.body("The project has 74 automated tests (Table 5.7) that run in about fifteen seconds. They check the data "
           "pipeline (shapes, absence of NaNs, a temporal-leakage check, training-only scaler fitting), the "
           "architectural guarantees, the loss functions and metrics against hand-computed values, the real-time "
           "data parsing and window layout, the new model options, the conformal scores, the network retry logic "
           "and the simulated-PV model.")
    R.table(
        ["Test file", "Tests", "Covers"],
        [
            ["test_phase1_pipeline.py", "9", "Window shapes, no NaNs, leakage check, scaler fitting, regime coverage, gap rejection"],
            ["test_phases_2_5.py", "19", "Three structural guarantees, output ranges, σ_expert detachment, gradient health"],
            ["test_losses_and_metrics.py", "14", "Hand-computed losses and metrics, pinball asymmetry, Winkler trade-off"],
            ["test_integration.py", "8", "Real training run, loss decrease, gate non-collapse, checkpoint round-trip"],
            ["test_realtime_sources.py", "5", "Solar geometry, weather parsing and alignment, live window layout"],
            ["test_residual_gate.py", "3", "Gate starts exactly uniform; δ receives gradient; default model unchanged"],
            ["test_last_step_skip.py", "3", "Shortcut is a no-op at start and receives gradient"],
            ["test_accuracy_options.py", "3", "Metadata inputs, extra loss term, option validation"],
            ["test_conformal_scores.py", "2", "Per-step scores restore coverage where window averages do not"],
            ["test_http_retry.py", "3", "Truncated downloads are retried, large requests split, 4xx not retried"],
            ["test_simulated_pv.py", "5", "No output at night, clipping, temperature effect, soiling, region flagged"],
            ["Total", "74", ""],
        ],
        "Automated test suite (74 tests, all passing).", widths=[5.2, 1.6, 8.8], size=9.5, align=[L_, C_, L_])
    R.h3("5.2.2", "Experimental Protocol")
    R.body("Every reported model was trained from several random starts (seeds 42, 43 and 44; the LSTM also 45, 46 and "
           "47 because of its instability). We report the mean and the sample standard deviation. The test set is "
           "never used for model selection: early stopping and calibration use the validation set. Hyper-parameters "
           "were chosen on Yorkshire, and the second region was run afterwards without any change.")
    R.h3("5.2.3", "Validity Checks")
    R.bullets([
        "**No look-ahead.** Only measured values up to the issue time are inputs; the weather forecasts for the "
        "next eight hours are inputs because they would be available at that time in live use.",
        "**Chronological splitting** with no shuffling and scalers fitted on training data only.",
        "**Reproduction check.** The LightGBM baseline run through the multi-seed tool with seed 42 reproduced the "
        "earlier single-run result exactly, confirming that the tool does not change the baselines.",
        "**Out-of-sample backtest** on 1–16 October 2026, after the end of the training period, using only data "
        "available at each issue time (Section 7.7).",
    ])
    R.h3("5.2.4", "Defects Found and Corrected")
    R.body("Testing and analysis revealed several defects and wrong conclusions in earlier versions of the work. They "
           "are listed in Table 5.8 because correcting them changed the results and the conclusions of this report.")
    R.table(
        ["Finding", "Consequence", "Correction"],
        [
            ["Early data source mis-scaled PV units and capacity", "Peaks clipped; unrealistic targets", "Convert Wh to kW, use the real capacity per row"],
            ["Weather input stopped at the issue time", "Model never saw the forecast for the hours it predicts", "Weather now spans history plus the 8-hour horizon"],
            ["Solar elevation ignored longitude and the equation of time", "Several hours of error away from Greenwich", "NOAA formulation in src/utils/solar.py"],
            ["Two ablations were mis-implemented", "One crashed; one did not disable what it claimed", "Fixed and re-run"],
            ["Learned gate raised error and variance", "nRMSE 4.39 ± 0.20 against 4.12 for uniform averaging", "Residual gate starting at uniform routing"],
            ["Single-seed comparisons", "An LSTM run gave 9.19% where other seeds gave 3.7%", "All comparisons use several seeds"],
            ["Calibration used a window average of signed violations", "Adjustments near zero or negative; volatile regime got narrower", "Per-step scores"],
            ["Night points counted in coverage", "Coverage looked like 0.80 while daytime coverage was 0.67", "Daytime-only calibration and reporting"],
            ["Truncated downloads on this network", "Pipeline crashed on corrupted JSON", "Retry plus automatic request splitting"],
            ["Dashboard demo ignored the shortcut; routing text was inaccurate", "Misleading demonstration", "Demo recomputes the full path; wording corrected"],
        ],
        "Defects found during testing and how they were corrected.", widths=[5.3, 5.5, 4.8], size=9, align=[L_, L_, L_])
