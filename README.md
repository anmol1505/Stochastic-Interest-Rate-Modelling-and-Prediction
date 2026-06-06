# Stochastic Interest Rate Modelling and Prediction
## Cox-Ingersoll-Ross Model: Implementation, Calibration, and Jump-Diffusion Extension
### Finance Club, IIT Roorkee — Open Projects 2026

---

**Author:** Anmol Seth  
**Enrollment No.:** 24117015

---

## What This Project Does

This notebook implements a complete stochastic short-rate modelling pipeline on real historical bond yield data. The workflow has five stages:

1. **Data Engineering** — robust preprocessing with outlier detection
2. **Base CIR** — exact MLE calibration via non-central chi-square transition density
3. **Prediction Challenge** — full yield curve reconstruction from the 3M rate alone
4. **Extension: CIR Jump-Diffusion** — Duffie-Pan-Singleton framework with compound Poisson jumps
5. **Critical Analysis** — structural limitations, regime analysis, failure modes

> **Prediction constraint**: only the 3M yield is used as test-time input; all longer maturities are held-out actuals compared against model predictions.

---

## Repository Structure

```
├── Stochastic_Interest_Rate_Modelling_and_Prediction.ipynb
├── train_data.csv
├── test_data.csv
├── test_data_3M.csv
└── README.md
```

---

## Dataset

The data contains daily zero-coupon bond yields across 9 maturity tenors:

| Column | Maturity |
|--------|----------|
| `ZC025YR` | 3 Months |
| `ZC050YR` | 6 Months |
| `ZC075YR` | 9 Months |
| `ZC100YR` | 1 Year |
| `ZC200YR` | 2 Years |
| `ZC500YR` | 5 Years |
| `ZC1000YR` | 10 Years |
| `ZC2000YR` | 20 Years |
| `ZC3000YR` | 30 Years |

The underlying asset and geographical region are undisclosed — the model relies purely on the mathematical relationships in the data.

The three files serve distinct roles:
- `train_data.csv` — full yield curve (all 9 maturities), used for MLE and λ calibration
- `test_data.csv` — yield curve actuals, used as held-out ground truth for evaluation
- `test_data_3M.csv` — only the 3M column, the **sole permitted test-time model input**

---

## Section A — Data Preprocessing

The raw data is intentionally noisy. A four-step automated cleaning pipeline handles every data quality issue:

| Step | Issue | Treatment | Rationale |
|------|-------|-----------|-----------|
| 1 | Whitespace in column names | `str.strip()` | Common artefact from Excel-exported CSVs |
| 2 | Missing values / NaNs | Forward-fill → dropna | Holiday/weekend carry-forward — standard Bloomberg convention |
| 3 | Outliers | 21-day rolling z-score, 4σ threshold, linear interpolation | 4σ is deliberately conservative — preserves genuine 50bp hike days |
| 4 | Non-positive rates | Clip → forward-fill | Required for the CIR square-root diffusion to remain well-defined |

**Key observations from exploratory analysis:**

The training set (2016–2024) spans two distinct macro regimes — a prolonged near-zero rate era (2016–2022) and a rapid tightening cycle (2022–2024). The test set (2024–2026) begins near the peak of that tightening cycle and captures the early plateau/easing phase. This regime discontinuity is the central challenge for any statically-calibrated model.

The ACF of daily Δr_t shows no significant autocorrelation beyond lag 1, consistent with the Markov property required by the CIR SDE — empirically validating the model's theoretical foundation for this dataset.

---

## Section B — CIR Model: Theory and Calibration

### The SDE

The Cox-Ingersoll-Ross (1985) model describes the instantaneous short rate $r_t$ under the physical measure $\mathbb{P}$ via:

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

The square-root diffusion has two crucial properties over simpler models like Vasicek:
- **Guaranteed positivity** — rates cannot cross zero provided the **Feller condition** $2\kappa\theta \geq \sigma^2$ holds
- **State-dependent volatility** — rate volatility scales with $\sqrt{r_t}$, matching the empirical fact that volatility is not constant

### Bond Pricing

$$P(t,T) = A(\tau)\,e^{-B(\tau)\,r_t}, \qquad \tau = T - t$$

$$y(\tau) = \frac{B(\tau)\,r_t - \ln A(\tau)}{\tau}$$

### Two-Tier Calibration: Physical vs Risk-Neutral Measure

A common mistake is calibrating entirely from time-series data and using those parameters for bond pricing. This ignores a fundamental distinction:

- **Physical measure $\mathbb{P}$** — how rates *actually evolve*. Calibrated from time-series via MLE.
- **Risk-neutral measure $\mathbb{Q}$** — the pricing measure. Calibrated from cross-sectional yield data.

They are linked via the market price of risk $\lambda$:

$$\kappa^* = \kappa + \lambda, \qquad \theta^* = \frac{\kappa\,\theta}{\kappa^*}$$

**Pipeline used:**
1. Exact MLE on the 3M time series → $(κ, θ, σ)$ under $\mathbb{P}$
2. Cross-sectional $\lambda$ fit on the last 126 training days → $(κ^*, θ^*)$ under $\mathbb{Q}$

### Why Exact MLE over Euler Discretisation?

The Euler approximation introduces an $\mathcal{O}(\sqrt{\Delta t})$ bias — on daily data, roughly a **6% downward bias in estimated σ**. Since σ enters the Feller condition and all bond pricing formulas, this is not negligible. The exact non-central chi-square density eliminates this bias entirely at negligible extra computational cost. There is no reason to accept Euler bias here.

---

## Section C — Prediction Challenge

Given calibrated risk-neutral parameters $(κ^*, θ^*, σ)$ and the observed 3M rate on each test day:

$$\hat{y}(\tau) = \frac{B^*(\tau)\,r_{3M} - \ln A^*(\tau)}{\tau}, \qquad \tau \in \{0.5,\ 0.75,\ 1.0,\ 2.0\}\text{ years}$$

**The base model achieves R² > 0.87**, clearing the 0.85 threshold. Per-maturity diagnostics reveal a systematic pattern:

| Maturity | RMSE | Structural Cause |
|----------|------|-----------------|
| 6M | ~6 bp | Short end closely anchored to 3M input |
| 9M | ~13 bp | Slope term premium partially enters |
| 1Y | ~19 bp | B(τ) saturation begins, θ* dominates |
| 2Y | ~35 bp | B(τ) nearly saturated; prediction driven almost entirely by θ* |

**The B(τ) saturation problem**: As τ → ∞, B(τ) → 2/γ — the model's sensitivity to the current short rate vanishes at long maturities. The 2Y prediction is therefore driven almost entirely by the risk-neutral long-run mean θ* rather than the input 3M rate. This is a fundamental structural constraint of single-factor affine models, not a calibration failure.

---

## Section D — Extension: CIR Jump-Diffusion

### Motivation

The base CIR model produces continuous sample paths — it cannot represent sudden, discontinuous jumps from central bank decisions, CPI/NFP surprises, or geopolitical shocks. Modelling these as large diffusion moves inflates σ and distorts the continuous-time dynamics. The jump-diffusion approach separates the two sources of variation cleanly.

### The SDE

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t + J\,dN_t$$

where $dN_t$ is a Poisson process with intensity $\lambda_J$ (jumps/year) and $J \sim \mathcal{N}(\mu_J, \sigma_J^2)$ — following Duffie, Pan and Singleton (2000). Six parameters: $(κ, θ, σ, λ_J, μ_J, σ_J)$.

### Transition Density

The exact density conditions on the number of jumps $n$ in $[t, t+\Delta t]$:

$$p(r_{t+\Delta t}\,|\,r_t) = \sum_{n=0}^{N_{\max}} \frac{e^{-\lambda_J\Delta t}(\lambda_J\Delta t)^n}{n!} \cdot p_{\text{CIR}}\!\left(r_{t+\Delta t} - n\mu_J\,\Big|\,r_t;\, \kappa, \theta, \tilde{\sigma}_n\right)$$

The sum is truncated at $N_{\max} = 8$ — negligible error since $\lambda_J \Delta t \approx 0.017$.

### Bond Pricing with Jumps

The model retains the exponential-affine form $P = A_{JD}(\tau)\,e^{-B(\tau)\,r_t}$ where $B(\tau)$ is **identical to base CIR** and $\ln A_{JD}$ picks up a jump compensator integral:

$$\ln A_{JD}(\tau) = \ln A_{CIR}(\tau) + \lambda_J \int_0^{\tau} \left[\mathbb{E}\!\left[e^{-B(u)J}\right] - 1\right] du$$

### Calibrated Jump Parameters

- **Jump intensity $\hat{\lambda}_J \approx 4.2$ jumps/year** — roughly one jump event every 12 weeks, consistent with the FOMC meeting frequency and major macro release cadence
- **Mean jump size $\hat{\mu}_J$** — near-zero or slightly negative, reflecting that upward and downward shocks are roughly balanced with a mild downward bias from flight-to-quality episodes
- **Likelihood ratio test stat >> 7.81** (chi-square critical at 5%, 3 df) — the jump process is statistically necessary, not noise

### Results

The JD extension improves RMSE at every maturity. Gains are largest at **9M and 1Y** — the intermediate maturities where the jump compensator is most active. The 2Y improvement is smaller because B(2.0) is near saturation and the prediction is dominated by θ*.

| Maturity | Base CIR RMSE | CIR-JD RMSE |
|----------|--------------|-------------|
| 6M | ~6.1 bp | ~5.7 bp |
| 9M | ~13.0 bp | ~12.4 bp |
| 1Y | ~19.4 bp | ~18.6 bp |
| 2Y | ~34.6 bp | ~33.2 bp |

---

## Section E — Critical Analysis

### Structural Limitations

**One-factor restriction — the binding constraint**: Even in the JD model, every yield remains a deterministic function of $r_t$. Perfect instantaneous correlation across all maturities is a strong and empirically violated assumption. In practice, the yield curve has at least three independent factors (level, slope, curvature). RMSE systematically grows with maturity because the 2Y yield carries slope and curvature information that cannot be recovered from the 3M rate alone in a one-factor framework.

**Physical κ identification**: The near-unit-root κ estimate is a well-documented artefact of MLE on short-rate time series (Hamilton & Wu 2012). For observation windows of 5–10 years, many different κ values produce nearly identical empirical distributions, making the likelihood nearly flat. True identification requires decades of data, Bayesian priors, or a Kalman filter over the cross-section.

**Static jump parameters**: Jump intensity and size are calibrated once and held fixed. In reality, λ_J is countercyclical — higher near central bank meeting dates, lower in quiet periods.

### Practical Suitability

| Use case | CIR-JD suitability | Key limitation |
|----------|-------------------|----------------|
| Short-horizon bond pricing (<2Y) | Moderate | Static jump params lag current vol regime |
| ALM / pension liability matching (>10Y) | Low | One-factor insufficient for duration hedging |
| Interest rate options (caps, swaptions) | Moderate | Jump vol adds option richness; tractable pricing |
| Stress testing / tail risk | Good | Jump process explicitly models discontinuous shocks |
| Regulatory VaR (daily, short horizon) | Good | Heavy tails correctly modelled |

---

## How to Run

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/165uvrYS3UT38Nq22Xg3_mI7hFt-oNoOp)

1. Click the badge above or [open directly in Colab](https://colab.research.google.com/drive/165uvrYS3UT38Nq22Xg3_mI7hFt-oNoOp)
2. Upload all three CSV files to your Colab session (or mount Google Drive)
3. Run **Runtime → Run all**

### Local
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy matplotlib scipy scikit-learn
jupyter notebook Stochastic_Interest_Rate_Modelling_and_Prediction.ipynb
```

---

## References

1. **Cox, J.C., Ingersoll, J.E., Ross, S.A.** (1985). *A Theory of the Term Structure of Interest Rates.* Econometrica, 53(2), 385–407.
2. **Duffie, D., Pan, J., Singleton, K.** (2000). *Transform Analysis and Asset Pricing for Affine Jump-Diffusions.* Econometrica, 68(6), 1343–1376.
3. **Merton, R.C.** (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous.* Journal of Financial Economics, 3(1–2), 125–144.
4. **Lee, S., Mykland, P.A.** (2008). *Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics.* Review of Financial Studies, 21(6), 2535–2563.
5. **Das, S.R.** (2002). *The Surprise Element: Jumps in Interest Rates.* Journal of Econometrics, 106(1), 27–65.
6. **Johannes, M.** (2004). *The Statistical and Economic Role of Jumps in Continuous-Time Interest Rate Models.* Journal of Finance, 59(1), 227–260.
7. **Hamilton, J.D., Wu, J.C.** (2012). *Identification and Estimation of Gaussian Affine Term Structure Models.* Journal of Econometrics, 168(2), 315–331.
8. **Piazzesi, M.** (2005). *Bond Yields and the Federal Reserve.* Journal of Political Economy, 113(2), 311–344.

---

*Finance Club, IIT Roorkee — Open Projects 2026*
