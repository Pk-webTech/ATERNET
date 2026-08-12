# ATERNET architecture

**Adaptive Temporal Expert Routing Network for Probabilistic Solar Power Forecasting**

This document is the single reference for how the five phases fit together,
what each design decision buys, and — equally important — where the design's
assumptions are load-bearing enough that they are enforced by tests.

## 1. System diagram

```
                    X_seq (B,48,10)   X_nwp (B,48,4)   X_sigma (B,48,6)
                          |                 |                |
                    ┌─────▼─────────────────▼────────────────▼─────┐
                    │  PHASE 2 — TemporalBackbone                  │
                    │  patchify(8/4) → 11 tokens → +pos_enc        │
                    │  PV patches ──cross-attend──> NWP patches    │
                    │  3-layer pre-norm transformer                │
                    └──┬──────────────┬─────────────────┬──────────┘
                       │              │                 │
              h_expert (B,128)   h_nwp (B,32)      h_sigma (B,32)
                       │              │                 │
        ┌──────────────▼───────────┐  │                 │
        │  PHASE 3 — ExpertPool    │  │                 │
        │  dawn_ramp   clear_peak  │  │                 │
        │  overcast    volatile    │  │                 │
        │  (all four, densely)     │  │                 │
        └──┬──────────────────┬────┘  │                 │
           │                  │       │                 │
    stacked (B,4,16)   sigma_expert (B,)  [DETACHED]    │
           │                  │       │                 │
           │        ┌─────────▼───────┴─────────────────▼─────┐
           │        │  PHASE 4 — AdaptiveRouter               │
           │        │  sigma_nwp = mean(spread cols)          │
           │        │  sigma_expert_cal = softplus(s)·sigma   │
           │        │  log1p → BatchNorm → convex combine     │
           │        │    sigma_total = α·σ_nwp + (1-α)·σ_exp  │
           │        │  gate = softmax(logits / T(sigma_total))│
           │        └────────────┬────────────────────────────┘
           │                     │
           └──────► ŷ_routed = Σ_k gate_k · ŷ_k  (B,16)
                                 │
                    ┌────────────▼─────────────────────────────┐
                    │  PHASE 5 — ProbabilisticDecoder          │
                    │  p50 = clamp(ŷ_routed + tanh(MLP)·0.1)   │
                    │  width = softplus(base + softplus(γ)·σ)  │
                    │  P10 = clamp(p50 - lower, 0, 1)          │
                    │  P90 = clamp(p50 + upper, 0, 1)          │
                    └────────────┬─────────────────────────────┘
                                 │
                          P10 ≤ P50 ≤ P90  (B,16)
```

## 2. The three guarantees the architecture makes

These are the claims that distinguish this design from "an MLP with extra
steps". Each is enforced structurally and verified by a named test.

| Guarantee | Mechanism | Test |
|---|---|---|
| **Quantiles never cross** | P10/P90 are P50 ∓ a softplus (≥0) half-width, so ordering holds before clamping; clamping to [0,1] cannot reorder them because P50 is already in [0,1] | `test_non_crossing_guaranteed`, `test_non_crossing_holds_on_real_data`, plus a re-check on every evaluation run |
| **Intervals widen with uncertainty** | γ passes through softplus, so ∂width/∂σ_total ≥ 0 everywhere. σ_total is *detached* on the MLP context path so the network cannot learn a compensating negative path | `test_width_is_monotone_non_decreasing_in_sigma` |
| **Routing softens under uncertainty** | T = exp(clamp(softplus(β)·σ_total)); softplus keeps β>0 so higher σ can only raise T (more uniform routing) | `test_temperature_increases_with_uncertainty` |

## 3. Why interval-width rather than dynamic quantile movement

The original design moved the quantile level τ at inference. Pinball loss is
a proper scoring rule **only for a τ fixed in advance**; moving τ invalidates
the guarantee that minimizing the loss recovers the true quantile. The design
was rebuilt around a fixed-τ pinball loss over a median-plus-non-negative-width
parameterization, which keeps the proper-scoring-rule property *and* upgrades
non-crossing from "penalized" to "impossible". See `src/losses/pinball_loss.py`.

## 4. Why σ_expert is detached

σ_expert measures how much the four experts disagree, and it feeds the gate.
If gradients flowed from the gate back into the experts through σ_expert, the
experts would have a direct incentive to collapse toward agreement — looking
confident rather than being accurate. Detaching makes σ_expert a pure
measurement. Verified by `test_sigma_expert_is_detached`.

## 5. The composite objective

```
L = 1.00 · pinball(P10,P50,P90)     ← the deliverable; proper at each fixed τ
  + 0.05 · mean(lower + upper)      ← sharpness; without it, coverage is trivially gamed by wide bands
  + w(t) · Σ_k expert_k loss        ← keeps all four experts learning (w: 1.0 → 0.3 after warmup)
  + 0.01 · load_balance(gate)       ← guards against gate collapse
```

The dense expert term is the non-obvious one. The routed forecast only
backpropagates into expert *k* in proportion to its gate weight, so an expert
the gate currently distrusts receives almost no gradient and can never improve
enough to earn trust back — a self-reinforcing deadlock. Training every expert
densely against the target in parallel breaks it. The weight anneals down so
that, once the experts are competent, the routed objective dominates and the
model optimizes the forecast it actually emits.

Model selection uses validation **pinball** loss, not the composite total,
because the composite's expert-term weight changes across epochs — comparing
composite totals between epoch 4 and epoch 20 compares two different objectives.

## 6. Known limitations

Stated plainly, because a reviewer will find them anyway:

1. **The data is synthetic.** No live PV telemetry or ECMWF access was
   available. The generators are physically grounded (solar geometry,
   PVWatts-style DC model, AR(1) cloud attenuation, spread correlated with
   volatility) but the reported metrics measure performance on a simulation,
   not on a real site. Every acquisition module is isolated behind a fixed
   output schema so swapping in real data touches only `download_*.py`.

2. **Regime labels are assigned at forecast issue time**, while the target
   spans the following 4 hours. A window labelled "overcast" frequently clears
   within the horizon. This is why the overcast expert's output cap had to be
   raised from 0.40 to 0.80 — see `src/phase3/overcast_expert.py`. Per-regime
   metrics should be read as "conditioned on issue-time conditions", not
   "conditioned on horizon conditions".

3. **σ_total is relative, not absolute.** Phase 4 BatchNorms each source, so
   σ_total is centred on the *training* average, not on a physical uncertainty
   scale. The "T = 1 at average uncertainty" reading is relative to training
   data. It should never be reported as a calibrated uncertainty in physical units.

4. **The volatile regime is thinly populated** (≈500 train / 36 test windows).
   Its per-regime metrics carry wide error bars, and the volatile expert has
   the largest capacity of the four trained on the least data — the ratio is
   defensible on pattern-space-coverage grounds but is not empirically validated
   at this sample count.

5. **Three quantile levels only.** The reported CRPS is a coarse
   three-point approximation and is labelled as indicative, not headline.
