# TACO Finance Experiments — ICAIF 2026

Three experiments applying the TACO latent-cost transfer framework to financial markets.

## Quick start

```bash
# from the dualwcfm/ directory
python -m icaif.run --exp risk_factors   # E3 (fastest, D=5)
python -m icaif.run --exp yield_curve    # E2 (D=10)
python -m icaif.run --exp iv_surface     # E1 (D=10)
python -m icaif.run --all                # all three

# With W&B logging
python -m icaif.run --all --wandb-project TACO-Finance --wandb-entity <you>

# Skip retraining (load saved models)
python -m icaif.run --exp iv_surface --no-retrain
```

## Experiments

### E1 — IV Surface Regime Transfer (`iv_surface`)
- **Setup**: Heston model, daily IV surface evolution (10 points: 5 strikes × 2 maturities).
- **Reference**: Calm Heston (θ=0.04, σ_ATM ≈ 20%), 3000 coupling days.
- **Transfer target**: Stressed Heston (θ=0.09, σ_ATM ≈ 30%), 800 days.
- **Shared cost**: κ, ξ, ρ identical across regimes — only the variance level shifts.
- **Key plots**: `smile_comparison.pdf`, `surface_heatmap.pdf`, `coupling_evolution.pdf`.

### E2 — Yield Curve Dynamics Transfer (`yield_curve`)
- **Setup**: Nelson–Siegel 3-factor model, daily (curve_today, curve_tomorrow) coupling (10 tenors).
- **Reference**: ZIRP epoch, 2500 days (~2015–2019 equivalent).
- **Transfer target**: Hiking epoch, 600 days (~2022–2023 equivalent).
- **Shared cost**: Factor loadings λ identical — only factor levels (β₀, β₁, β₂) shift.
- **Key plots**: `curve_comparison.pdf`, `pca_distribution.pdf`, `correlation_heatmaps.pdf`.

### E3 — Cross-Regime Risk Factor Scenario Generation (`risk_factors`)
- **Setup**: 5 risk factors (equity, vol, rates, FX, credit), daily transition coupling.
- **Reference**: Normal VIX regime, 2000 days.
- **Transfer target**: Stressed VIX regime, 200 days (data-scarce).
- **Shared cost**: Correlation matrix R — only marginal vols scale up 2-3×.
- **Key plots**: `joint_scatter.pdf`, `correlation_comparison.pdf`, `marginal_distributions.pdf`, `portfolio_var.pdf`.

## File structure

```
icaif/
├── run.py                      # Entry point
├── data/
│   ├── heston.py               # Heston simulation + IV surface (Bergomi–Gatheral approx.)
│   ├── yield_curve.py          # Nelson–Siegel synthetic + optional FRED real data
│   └── risk_factors.py         # Multivariate Gaussian synthetic + optional yfinance
├── distributions.py            # FinancialCouplingDistribution, FinancialMarginalDistribution
├── experiments/
│   ├── iv_surface.py           # ExperimentSpec factory for E1
│   ├── yield_curve.py          # ExperimentSpec factory for E2
│   └── risk_factors.py         # ExperimentSpec factory for E3
├── visualizer.py               # FinanceVisualizer + three mode-specific subclasses
└── plots/
    ├── theme.py                # Consistent colour palette + rcParams
    ├── iv_surface.py           # IV smile, heatmap, coupling evolution
    ├── yield_curve.py          # Curve comparison, PCA, correlation heatmaps
    └── risk_factors.py         # Joint scatter, correlation comparison, VaR bars
```

## Output structure (per experiment)

```
results/icaif/<exp_name>/
├── metrics.csv              # Standard TACO metrics (SW2, MMD, ESS, …) per iteration
├── finance_metrics.csv      # Domain-specific metrics (IV RMSE, Frobenius corr err, VaR)
├── finance/
│   ├── *.pdf                # Publication-quality figures (see above)
│   └── finance_NNNN.png     # Per-checkpoint domain plot (logged to W&B)
├── coupling/                # Standard TACO coupling snapshots
├── marginals/               # Marginal distribution evolution
├── weights/                 # Importance weight histograms
└── models/                  # Saved neural network weights
```

## Design notes

All three experiments follow the same pattern as the existing singlecell experiment:
- Precomputed numpy arrays → `FinancialCouplingDistribution` / `FinancialMarginalDistribution`
- Standard TACO training loop via `Trainer` + `MixtureMethod` (particle_ratio=0.2)
- Custom `FinanceVisualizer` subclass adds domain plots on top of standard TACO metrics
- W&B logging via the existing `wandb_logger` infrastructure

No core TACO code is modified.
