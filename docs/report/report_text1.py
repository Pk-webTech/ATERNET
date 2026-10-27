"""Abstract and Chapters 1-3."""

from report_data import d, fmt, FIG
from report_lib import C_, J, L_, R_

TITLE = "ATERNET: Adaptive Temporal Expert Routing Network for Probabilistic Solar Power Forecasting"


def abstract_paragraphs():
    y, sw = d.Y[d.A], d.SW[d.A]
    lg, pt = d.Y["LightGBM quantile"], d.Y["PatchTST quantile"]
    sl, sp = d.SW["LightGBM quantile"], d.SW["PatchTST quantile"]
    cy, cs = d.cal["yorkshire"], d.cal["south_west"]
    return [
        "Solar photovoltaic (PV) generation depends on weather that cannot be predicted exactly, so grid operators "
        "need forecasts that state not only how much power to expect but also how sure the forecast is. This project "
        "presents ATERNET (Adaptive Temporal Expert Routing Network), a deep-learning model that forecasts regional "
        "solar generation for the next eight hours at 30-minute resolution and returns three quantiles (P10, P50 and "
        "P90) that together form an 80% prediction interval. The model combines a PatchTST-style Transformer "
        "backbone, a pool of four weather-regime experts, an uncertainty-aware router that fuses the disagreement "
        "between numerical weather prediction (NWP) models with the disagreement between the experts, and a "
        "probabilistic decoder whose interval width grows with that uncertainty.",

        "ATERNET was trained and tested on measured regional PV output from Sheffield Solar PV_Live for two English "
        "regions (Yorkshire and South West England, January 2023 to September 2026) together with forecasts from five "
        "NWP models obtained through Open-Meteo. All comparisons use several random training runs, and a difference "
        "is reported only when it exceeds the run-to-run variation. Our first design, with a learned routing gate, "
        f"was less accurate and far less stable than a plain average of the experts (nRMSE {d.dev[0][2]:.2f} ± "
        f"{d.dev[0][3]:.2f}% against {d.dev[1][2]:.2f}%). A residual gate that starts as uniform routing, a "
        "zero-initialised linear shortcut from the latest observation and exponential averaging of the weights fixed "
        f"this and brought the error to {fmt(y, 'nrmse_pct', 2)}%, level with LightGBM ({fmt(lg, 'nrmse_pct', 2)}%) "
        f"and clearly better than PatchTST ({fmt(pt, 'nrmse_pct', 2)}%); its forecast ranges obtained a better "
        f"Winkler score than both. On South West England, with no re-tuning, ATERNET was ahead of LightGBM "
        f"({fmt(sw, 'nrmse_pct', 2)}% against {fmt(sl, 'nrmse_pct', 2)}%) and PatchTST ({fmt(sp, 'nrmse_pct', 2)}%).",

        "Rigorous checking also corrected our own early results. Coverage statistics that include night-time hours "
        f"overstated how well the ranges held; in daylight the raw ranges contained the real value only "
        f"{cy['picp_daytime_before'] * 100:.0f}% of the time instead of 80%. After fixing the conformal calibration "
        f"step and applying it to daytime points, daytime coverage reached {cy['picp_daytime_after'] * 100:.0f}% and "
        f"{cs['picp_daytime_after'] * 100:.0f}% on the two regions. Ablation studies showed that only the dense expert "
        "loss measurably improved the results; the adaptive gate, the temperature control and regime specialisation "
        "did not, and we report this openly.",

        "Because no open measured PV record exists for the VIT Vellore area, we also built a clearly labelled "
        "simulated demonstration for Vellore, driven by real weather data. The work is delivered with an automated "
        "test suite of 74 tests and an interactive dashboard that explains every result in plain language.",
    ]


def chapter1(R):
    R.chapter_heading(1, "Introduction")
    R.h2("1.1", "Background")
    R.body("Solar photovoltaic (PV) power is among the fastest-growing sources of electricity worldwide. Unlike a "
           "conventional power station, a PV plant cannot be dispatched on demand: its output follows the sun and is "
           "cut sharply by clouds. When the share of solar power in a grid is small this variability is absorbed "
           "without difficulty, but as the share grows, system operators must anticipate how much solar energy will be "
           "available in the next minutes, hours and days in order to schedule other generation, keep reserves ready, "
           "trade energy and maintain the balance between supply and demand [@lo2025,zhao2026twostage].")
    R.body("Forecasts of PV power are produced from three kinds of information: recent measurements of the plant's "
           "own output, satellite and sky observations, and numerical weather prediction (NWP), in which "
           "physics-based atmospheric models are run on supercomputers to predict irradiance, cloud cover and "
           "temperature [@zhao2026twostage]. For horizons of a few hours, the most useful forecasts combine the latest "
           "measurements with NWP output, because the measurements describe what is happening now while the NWP "
           "describes what the atmosphere is about to do.")
    R.body("A single number is not enough for decisions. If a forecast says 300 MW but the true value could easily lie "
           "anywhere between 150 MW and 450 MW, an operator will plan very differently than if the plausible range is "
           "280 to 320 MW. For this reason the research community has moved from point forecasts to probabilistic "
           "forecasts, which describe the whole distribution of possible outcomes [@lo2025,lyu2024]. A common "
           "and practical form is a prediction interval, for example the range between the 10th and 90th percentiles "
           "(P10 and P90) together with a median (P50). Such an interval is useful only if it is reliable (the real "
           "value falls inside it about as often as promised) and sharp (it is no wider than necessary).")
    R.h2("1.2", "Motivations")
    R.body("Modern deep-learning models, and in particular Transformer networks, have improved forecasting accuracy in "
           "many domains, but three practical problems remain for solar forecasting. First, the accuracy of a model "
           "depends strongly on the weather situation: a clear sunny midday, a ramp at sunrise, a uniformly cloudy day "
           "and a day of broken clouds behave very differently, and a single network tends to be averaged over them. "
           "Second, the uncertainty of a forecast has two different sources, namely disagreement between weather "
           "models and the model's own uncertainty, and most systems either ignore one of them or learn a single "
           "opaque confidence value. Third, results in this field are often reported from a single training run and "
           "with statistics that look better than they are, which makes it hard to know which improvements are real.")
    R.body("These observations motivated us to design a forecaster that (i) separates the weather situations into "
           "specialised experts, (ii) combines the two sources of uncertainty explicitly and uses them to decide both "
           "how much to trust each expert and how wide to draw the forecast interval, and (iii) is evaluated "
           "honestly, with several random starts, strong baselines and an explicit separation of what the data do and "
           "do not support. We also wanted the work to be demonstrable for our own campus at VIT Vellore.")
    R.h2("1.3", "Scope of the Project")
    R.body("The project covers the complete pipeline for regional PV forecasting: data acquisition, cleaning and "
           "feature engineering, model design and training, comparison with three baseline models, calibration of the "
           "forecast intervals, an interactive dashboard and an automated test suite. Specifically:")
    R.bullets([
        "**Forecast task.** Predict the regional PV output for the next 16 half-hours (8 hours) as P10, P50 and P90, "
        "given the previous 24 hours of measured output and the weather forecasts for the past and the next 8 hours.",
        "**Data.** Measured half-hourly regional generation from Sheffield Solar PV_Live for the Yorkshire region "
        "(NPG Yorkshire) and for South West England (WPD South West), and five-model weather forecasts from "
        "Open-Meteo, from 1 January 2023 to 30 September 2026.",
        "**Evaluation.** Accuracy (nRMSE, nMAE, R²), interval quality (coverage, width, Winkler score), comparison "
        "with LightGBM, PatchTST and LSTM, ablation studies, multiple random seeds and an out-of-sample backtest on "
        "October 2026.",
        "**Demonstration for India.** A clearly labelled simulated plant at VIT Vellore, because no open measured PV "
        "record exists for the area.",
    ])
    R.body("The project does not cover forecasting for individual panels, horizons beyond eight hours, grid control or "
           "market bidding, and it makes no claim about real measured performance in India, for which we were unable "
           "to obtain data (Section 7.8).")


def chapter2(R):
    R.chapter_heading(2, "Project Description and Goals")
    R.h2("2.1", "Literature Review")
    R.body("We studied twenty recent papers on probabilistic forecasting of solar power and of other energy series "
           "(Table 2.1). They are grouped below by the idea they contribute, and each group is related to the design "
           "of ATERNET. Some of the papers concern other energy series (wind, load, curtailment, thermal energy); "
           "this is stated where it matters, because their methods can transfer but their results do not.")
    R.body("**Quantile-based probabilistic forecasting of PV.** Most probabilistic PV models output quantiles trained "
           "with the pinball loss [@koenker1978,gneiting2007]. Lo et al. [@lo2025] propose a quantile-regression "
           "procedure that skips training the separate upper- and lower-bound models when a site's output is stable. "
           "He et al. [@he2026mrdl] decompose the power series with adaptive variational mode decomposition, predict "
           "with a bidirectional LSTM with feature attention, and add an asymmetric monotonicity penalty so that "
           "quantiles do not cross; Huang et al. [@huang2026] likewise use a non-crossing joint quantile loss for a "
           "distributed PV cluster. Zhao et al. [@zhao2026twostage] feed multi-source NWP forecasts to a two-stage "
           "ensemble in which diverse base learners produce quantiles and an attention-based meta-learner combines them "
           "under a non-crossing quantile generation strategy. Phan et al. [@phan2024] combine a Transformer with "
           "lower-upper-bound estimation, XGBoost-based imputation of missing data and a GRU that corrects residual "
           "bias, and He et al. [@he2026gan] use a generative adversarial network to learn the joint distribution of "
           "measurements and NWP, with a multi-branch CNN-LSTM and quantile regression with kernel density estimation. "
           "In ATERNET the quantiles cannot cross by construction, because the interval is built from positive "
           "half-widths around the median, so no penalty term is needed.")
    R.body("**Dependence on the weather situation.** Forecast difficulty depends on the weather. Lyu and "
           "Eftekharnejad [@lyu2024] combine copula theory with an XGBoost classifier so that the dependence between "
           "meteorological variables changes with the weather type, and report gains of up to 60% under non-sunny "
           "conditions. Wang et al. [@wang2026joint] use a Transformer point forecast, an autoencoder that recognises "
           "fluctuation patterns and a Clayton copula for the joint distribution of forecast values and errors. Related "
           "ideas appear in other energy domains: TADNet [@ye2024] treats a series as a regular component plus a "
           "residual whose fluctuation interval is quantified, and Zhao et al. [@zhao2024] add Gaussian-process "
           "residual learning to a Transformer for multi-energy load forecasting. These works support treating "
           "different situations differently, which ATERNET does with its regime experts.")
    R.body("**Mixtures of experts and gating.** A mixture of experts splits a task between specialised sub-models and "
           "a gating network that combines them [@jacobs1991,shazeer2017]. Wang et al. [@wang2025cg] use a "
           "context-gated mixture of experts to weight LSTM, GRU, Transformer and tree-based learners for PV "
           "forecasting, with a gate driven by input features, local prediction variance and historical residuals, "
           "followed by a residual corrector and a conformal interval. Heidrich et al. [@heidrich2026] place a mixture "
           "of experts with SwiGLU feed-forward experts and an auxiliary load-balancing loss inside a patch-based "
           "Transformer for curtailment forecasting, and Seman et al. [@seman2026] use sparse expert Transformers "
           "with measures against expert collapse for thermal energy forecasting. Load-balancing penalties are used "
           "to keep all experts in use [@shazeer2017,heidrich2026]. These studies show that gating is attractive, "
           "but they do not settle whether a learned gate beats a plain average of the experts for regional PV, "
           "which the experiments in Chapter 7 test directly.")
    R.body("**Calibrated intervals.** Conformal prediction gives distribution-free intervals. Romano et al. "
           "[@romano2019] conformalise quantile regression, and Jensen et al. [@jensen2024] extend this to time series "
           "with a bootstrap ensemble of quantile regressors and asymmetric conformity scores, obtaining sharper "
           "valid intervals. Conformal steps are also part of the PV and curtailment frameworks of [@wang2025cg] "
           "and [@heidrich2026]. We use the quantile-regression conformity score of Romano et al. per weather "
           "regime (Section 5.1.7).")
    R.body("**Spatial and image-based methods.** Another group of recent papers uses spatial structure or "
           "imagery. Graph networks model the dependence between PV sites: Wang et al. [@wang2026dgn] with a dynamic "
           "graph and a loss that aligns both the shape and the amplitude of the forecast, Lin et al. [@lin2025] with a "
           "convolutional graph network and parameter transfer on 127 plants, Huang et al. [@huang2026] with "
           "spatial-temporal attention, and Song et al. [@song2025] with random coarse-graph attention that also "
           "handles missing data across 1,649 sites. Image-based work predicts cloud motion: Rastgoo et al. "
           "[@rastgoo2025] use a diffusion model on sky-image sequences with an evidence-lower-bound loss, and Wang "
           "et al. [@wang2026clouddiff] a diffusion model on satellite images with a neural stochastic differential "
           "equation, whose image ensembles are combined with PV history and NWP. These methods need plant-level "
           "networks or image streams; ATERNET needs only the regional PV series and NWP forecasts. For wind power, "
           "Abedinia et al. [@abedinia2024] estimate the parameters of a spline quantile function with a CNN-LSTM "
           "trained with an adaptive weighted CRPS, an alternative to predicting fixed quantiles.")
    R.body("**Sequence models and boosted trees.** Recurrent networks, in particular the long short-term memory network "
           "(LSTM) [@hochreiter1997], were for years the standard deep model for time series. The Transformer "
           "[@vaswani2017] replaced recurrence by attention, and PatchTST [@nie2023] adapted it to forecasting by "
           "splitting each series into short overlapping patches that are treated as tokens. We use a PatchTST-style "
           "encoder as the backbone of ATERNET and a vanilla PatchTST predicting quantiles as a baseline. LightGBM "
           "[@ke2017] is a fast gradient-boosting library that remains very competitive on tabular data; it does not "
           "model time explicitly, so features describing the recent past and the forecast weather must be supplied. "
           "Because of its strength we treat it as the principal baseline.")
    R.body("**Scoring and training stability.** An interval is assessed by its coverage (PICP), its width (PINAW) "
           "and by the Winkler score [@winkler1972], which adds a penalty proportional to the amount by which the "
           "observation lies outside the interval, so that a model cannot look good simply by drawing very wide "
           "intervals. ReZero [@bachlechner2021] multiplies each residual branch by a learnable scalar initialised to "
           "zero so that a network starts as the identity and only departs from it when that helps. Averaging the "
           "weights along the training trajectory [@polyak1992], today usually implemented as an exponential moving "
           "average (EMA), is a cheap way to obtain a smoother and often more accurate model. Both ideas proved "
           "decisive in our work (Section 7.4).")
    R.table(
        ["Work", "Approach (as stated in the paper)", "Relation to ATERNET"],
        [
            ["Wang et al. 2026 [@wang2026dgn]", "Dynamic graph network over distributed PV sites; shape-amplitude loss (dynamic time warping + MSE); quantile regression", "Needs several sites; ATERNET models one regional series"],
            ["Lo et al. 2025 [@lo2025]", "Quantile regression that skips training upper and lower bounds when output is stable; missing-data imputation", "ATERNET always outputs three quantiles"],
            ["He et al. 2026 [@he2026mrdl]", "Adaptive VMD, BiLSTM with feature attention, asymmetric monotonicity penalty against quantile crossing", "ATERNET prevents crossing by construction"],
            ["Wang et al. 2025 [@wang2025cg]", "Context-gated mixture of experts over LSTM, GRU, Transformer and tree learners; residual corrector; conformal interval; no NWP", "Closest gating idea; ATERNET gates on NWP spread and expert disagreement"],
            ["Zhao et al. 2026 [@zhao2026twostage]", "Two-stage ensemble with meta-learner on multi-source NWP; skip connections; non-crossing quantile generation", "Closest NWP use; ATERNET uses the spread between NWP models as uncertainty"],
            ["He et al. 2026 [@he2026gan]", "Similar-moment selection; GAN on measurements + NWP; multi-branch CNN-LSTM; quantile regression with KDE", "Data augmentation; ATERNET uses no generative model"],
            ["Heidrich et al. 2026 [@heidrich2026]", "Patch-based Transformer with SwiGLU mixture of experts, load-balancing loss and conformal inference (curtailment)", "Same building blocks, different series"],
            ["Lyu and Eftekharnejad 2024 [@lyu2024]", "Copula theory with an XGBoost weather classifier for changing dependencies", "Weather-dependent modelling, like our regimes"],
            ["Lin et al. 2025 [@lin2025]", "Dynamic convolutional graph network; parameter transfer; regional PV from 127 plants", "Needs plant-level data"],
            ["Huang et al. 2026 [@huang2026]", "Spatial-temporal attention and graph convolution; non-crossing joint quantile loss", "Cluster graph; ATERNET uses no graph"],
            ["Rastgoo et al. 2025 [@rastgoo2025]", "Diffusion model on sky-image sequences; ELBO loss; ultra-short term", "Needs sky images"],
            ["Seman et al. 2026 [@seman2026]", "Hierarchical VMD, architecture search, sparse expert Transformers (thermal energy)", "Different series; shares expert idea"],
            ["Phan et al. 2024 [@phan2024]", "Transformer-LUBE; XGBoost imputation; GRU residual correction; day-ahead PV with NWP", "Point-and-bounds design; ATERNET is quantile-based"],
            ["Jensen et al. 2024 [@jensen2024]", "Ensemble conformalized quantile regression with asymmetric conformity scores", "Related calibration; we use the CQR score per regime"],
            ["Song et al. 2025 [@song2025]", "Random coarse-graph attention and imputation across 1,649 PV sites", "Large-scale graph; ATERNET is single-region"],
            ["Wang et al. 2026 [@wang2026clouddiff]", "Diffusion-based satellite image prediction; neural SDE; multimodal ensemble with PV and NWP", "Needs satellite images"],
            ["Zhao et al. 2024 [@zhao2024]", "Hybrid-attention Transformer with Gaussian-process residual learning (multi-energy load)", "Residual modelling idea; different series"],
            ["Ye et al. 2024 [@ye2024]", "Regular component plus residual fluctuation interval; convolution with attention and gating", "Decomposition of the series; different series"],
            ["Wang et al. 2026 [@wang2026joint]", "Transformer, autoencoder fluctuation patterns and Clayton copula for forecast intervals", "Pattern-dependent intervals; copula-based"],
            ["Abedinia et al. 2024 [@abedinia2024]", "CNN-LSTM estimating a spline quantile function; adaptive weighted CRPS (wind power)", "Alternative to fixed quantiles; wind"],
        ],
        "The twenty papers studied and how each relates to ATERNET.",
        widths=[3.9, 7.2, 4.5], size=8.5, align=[L_, L_, L_])
    R.h2("2.2", "Research Gap")
    R.body("From these papers and from our preliminary experiments we identified the following gaps, which define the "
           "contribution of this project:")
    R.bullets([
        "**Uncertainty from disagreement.** Among the twenty papers, we could not find one that uses the disagreement "
        "between several NWP models, or between several experts, as an explicit uncertainty signal that also sets "
        "the width of the interval and the sharpness of a gate. Multi-source NWP is used as input in "
        "[@zhao2026twostage] and gating on prediction variance in [@wang2025cg], but not the two together.",
        "**Value of gating.** Mixtures of experts and gating are reported for PV, curtailment and thermal energy "
        "[@wang2025cg,heidrich2026,seman2026], but whether a learned gate **actually beats a plain average** of the "
        "experts, and whether it is stable across training runs, needs a direct test for regional PV.",
        "**Calibration where it matters.** PV series contain exact zeros at night, where a zero-width interval is "
        "trivially correct. Coverage should therefore be assessed, and conformal calibration fitted, on **daytime** "
        "points (Section 5.1.7); we found that all-hours coverage hides under-coverage in daylight.",
        "**Data requirements.** Graph and image methods [@wang2026dgn,lin2025,huang2026,song2025,rastgoo2025,"
        "wang2026clouddiff] need plant-level networks or image streams. A model that works from a regional "
        "aggregate and NWP forecasts is useful where these are not available.",
        "**Reliable comparison.** Differences of a few percent between models can be smaller than the run-to-run "
        "variation of training. We use several random starts, strong baselines and a second region, and call a "
        "difference real only when it exceeds that variation.",
        "**Regions without open data.** For areas such as the one around our campus, a clearly labelled "
        "**simulated demonstration** is needed that does not pretend to be a real validation.",
    ], numbered=True)
    R.h2("2.3", "Objectives")
    R.body("The objectives of the project, with the way each one is verified, are:")
    R.table(
        ["No.", "Objective", "How it is verified"],
        [
            ["O1", "Route between weather-regime experts using the combined uncertainty (context-gated routing).",
             "Gate behaviour (Fig. 7.8), ablations without the gate (Table 7.5)"],
            ["O2", "Make the width of the forecast interval depend on the uncertainty (uncertainty-conditioned decoding).",
             "Correlation of width and error (Fig. 7.9), width-ablation, safety check 2"],
            ["O3", "Train all parts jointly, keeping every expert learning (dense expert loss).",
             "Ablation without the dense expert loss (Table 7.5)"],
            ["O4", "Evaluate per weather regime and per forecast horizon.",
             "Tables 7.8 and Fig. 7.10"],
            ["O5", "Compare honestly with strong baselines over several random seeds and on a second region.",
             "Tables 7.1 and 7.3, Figures 7.1 and 7.2"],
            ["O6", "Deliver a tested, reproducible system and an understandable dashboard, including a simulated "
                   "demonstration for VIT Vellore.",
             "74 automated tests (Table 5.7), Chapter 6, Section 7.8"],
        ],
        "Project objectives and their verification.", widths=[1.2, 8.4, 6.0], size=9.5, align=[C_, L_, L_])
    R.h2("2.4", "Problem Statement")
    R.body("Given the measured generation of a PV region over the last 24 hours and multi-model weather forecasts for the "
           "last 24 hours and the next 8 hours, produce for each of the next 16 half-hours a median forecast and an "
           "80% prediction interval (P10 to P90) such that (a) the median is as accurate as the best available "
           "baseline, (b) the real value falls inside the interval about 80% of the time in daylight, (c) the "
           "interval is as narrow as possible, (d) the result is stable across random training starts, and (e) the "
           "system can be explained to a non-specialist.")
    R.h2("2.5", "Project Plan")
    R.body("The work was carried out in the phases shown in the plan below. The first model, with a data pipeline "
           "and a synthetic data source, was built in August 2026; real data were integrated from September; the "
           "main experiments, improvements and corrections took place in October 2026, followed by the dashboard, "
           "the second region, the simulated India demonstration and this report.")
    R.fig(FIG / "fig_2_1_project_plan.png", "Project plan and timeline (July to November 2026).", width_cm=15.2)
    R.table(
        ["Phase", "Main deliverable"],
        [
            ["Problem study, literature review", "Problem definition, design of the five-phase architecture"],
            ["Pipeline and model (Phases 1–5)", "Data pipeline, backbone, experts, router, decoder, first training run"],
            ["Real-data integration", "PV_Live and Open-Meteo sources, corrected units and capacity, longer weather input"],
            ["Baselines, ablations, improvements", "LightGBM/PatchTST/LSTM, multi-seed protocol, residual gate, shortcut, EMA"],
            ["Second region and India demo", "South West England experiment, simulated Vellore site"],
            ["Dashboard, tests, documentation", "Streamlit dashboard, 74 tests, this report"],
        ],
        "Phases of the project and their deliverables.", widths=[6.0, 9.6], size=9.5, align=[L_, L_])


def chapter3(R):
    R.chapter_heading(3, "Technical Specification")
    R.h2("3.1", "Requirements")
    R.h3("3.1.1", "Functional")
    R.table(
        ["ID", "Requirement"],
        [
            ["F1", "Download measured regional PV generation and installed capacity (PV_Live) for a chosen region and period."],
            ["F2", "Download multi-model weather forecasts (Open-Meteo) and align them to the PV time grid."],
            ["F3", "Clean the data, derive features (clear-sky index, time encodings, forecast-spread features, regime labels) and build 24 h to 8 h windows without leaking future information."],
            ["F4", "Train the ATERNET model jointly with early stopping, checkpointing and optional weight averaging."],
            ["F5", "Output P10, P50 and P90 for 16 steps with the guarantee P10 ≤ P50 ≤ P90."],
            ["F6", "Evaluate with nRMSE, nMAE, R², PICP, PINAW, Winkler score and skill against persistence, per regime and per horizon."],
            ["F7", "Train and evaluate the LightGBM, PatchTST and LSTM baselines under the same protocol."],
            ["F8", "Calibrate the intervals per weather regime on daytime validation points."],
            ["F9", "Run experiments over several random seeds and aggregate means and standard deviations."],
            ["F10", "Provide live forecast and backtest commands, and an interactive dashboard of the results."],
            ["F11", "Support a second region and a clearly flagged simulated site through configuration only."],
        ],
        "Functional requirements.", widths=[1.3, 14.3], size=9.5, align=[C_, L_])
    R.h3("3.1.2", "Non-Functional")
    R.table(
        ["ID", "Requirement", "How it is met"],
        [
            ["N1", "Reproducibility", "Fixed seeds, saved configuration and checkpoints, results written to files, one command per experiment"],
            ["N2", "Scientific validity", "Chronological split, scalers fitted on training data only, multi-seed reporting, results read from files"],
            ["N3", "Robustness", "Retry and automatic request-splitting for downloads; gap detection; automatic early stopping"],
            ["N4", "Isolation", "Environment variables move the data, outputs and checkpoints to a separate workspace so experiments cannot overwrite each other"],
            ["N5", "Understandability", "Dashboard in plain language with a glossary and colour-coded comparison"],
            ["N6", "Efficiency", "A full training run takes about 10 minutes on a laptop CPU"],
            ["N7", "Maintainability", f"Modular code (about {d.loc:,} lines of Python in src, config and tests) and 74 automated tests"],
            ["N8", "Honesty of reporting", "Wins are claimed only beyond ~2 seed standard deviations; simulated data are labelled everywhere"],
        ],
        "Non-functional requirements.", widths=[1.3, 3.8, 10.5], size=9.5, align=[C_, L_, L_])
    R.h2("3.2", "Feasibility Study")
    R.h3("3.2.1", "Technical Feasibility")
    R.body("All required components are available as mature open-source software (Python, PyTorch, pandas, "
           "LightGBM, Streamlit) and the data sources are public web services. The model has about 968,000 trainable "
           "parameters and the training set has about 11,500 windows, so a full training run fits comfortably on an "
           "ordinary laptop without a GPU. This was confirmed in practice: all experiments in this report, including "
           "more than 60 training runs, were executed on a laptop with an Intel Core i5-12450H processor and 7.7 GB of "
           "memory. The main technical risks were the quality of the open data (handled by validation and tests) and "
           "the memory limit of the laptop (handled by running one job at a time).")
    R.h3("3.2.2", "Economic Feasibility")
    R.body("The project uses no paid software, data or cloud services. Sheffield Solar PV_Live and Open-Meteo are free "
           "for research use, and every library is open source. The only cost is the author's time and the electricity "
           "of a laptop running for a few hours per experiment (Section 7.11). A deployment for an operator would add "
           "the cost of a data feed and hosting, but the model is small enough to run on a single inexpensive server "
           "or even on the operator's existing computers.")
    R.h3("3.2.3", "Social Feasibility")
    R.body("Better solar forecasts help grids integrate more renewable energy, reduce the need for fossil-fuel "
           "reserve generation and lower energy costs. Interval forecasts that are honest about uncertainty reduce the "
           "risk of over-reliance on a single number. The system works on aggregated regional data and uses no "
           "personal information. The main social risk is misuse of the simulated Indian demonstration as though it "
           "were measured evidence; for that reason it is labelled as simulated in the code, in the dashboard and in "
           "this report.")
    R.h2("3.3", "System Specification")
    R.h3("3.3.1", "Hardware Specification")
    R.table(
        ["Item", "Specification used"],
        [
            ["Processor", "Intel Core i5-12450H (8 cores, 12 threads)"],
            ["Memory", "7.7 GB RAM"],
            ["Graphics", "NVIDIA GeForce GTX 1650 present, but PyTorch was the CPU build; all training ran on the CPU"],
            ["Storage", "A few GB for cached downloads, datasets, checkpoints and results"],
            ["Network", "Internet access for data downloads and live forecasts"],
        ],
        "Hardware used for development and experiments.", widths=[4.0, 11.6], size=9.5, align=[L_, L_])
    R.h3("3.3.2", "Software Specification")
    R.table(
        ["Component", "Version / description"],
        [
            ["Operating system", "Windows 11 (10.0.26300)"],
            ["Language", "Python 3.11.9"],
            ["Deep learning", "PyTorch 2.5.1 (CPU) [@paszke2019]"],
            ["Gradient boosting", "LightGBM 4.7.0 [@ke2017]"],
            ["Data handling", "NumPy 2.2.6, pandas 2.2.3, PyArrow 22.0.0, SciPy 1.15.1, scikit-learn 1.8.0"],
            ["Plotting and dashboard", "Matplotlib 3.9.2, Plotly 6.1.0, Streamlit 1.50.0 [@streamlit]"],
            ["Testing", "pytest 9.1.1 (74 tests)"],
            ["Data services", "Sheffield Solar PV_Live API [@pvlive]; Open-Meteo forecast archive and ERA5 reanalysis API [@openmeteo,hersbach2020]"],
            ["Version control", f"Git ({d.commits} commits at the time of writing)"],
        ],
        "Software environment.", widths=[4.4, 11.2], size=9.5, align=[L_, L_])
