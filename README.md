# Stochastic Interest Rate Modelling and Prediction
### Cox-Ingersoll-Ross Model: Implementation, Calibration, and Jump-Diffusion Extension
**Finance Club, IIT Roorkee — Open Projects 2026**

---

**Author:** Anmol Seth  
**Enrollment No.:** 24117015

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Problem Statement](#core-problem-statement)
3. [Background & Theory](#background--theory)
4. [Repository Structure](#repository-structure)
5. [Datasets](#datasets)
6. [Workflow & Methodology](#workflow--methodology)
7. [Model Extensions](#model-extensions)
8. [Evaluation Criteria](#evaluation-criteria)
9. [Requirements](#requirements)
10. [How to Run](#how-to-run)
11. [Key Results & Analysis](#key-results--analysis)
12. [References](#references)

---

## Project Overview

Interest rates are the fundamental building blocks of the global financial system — they dictate the pricing of bonds, the valuation of derivatives, and the risk management strategies of institutional portfolios. This project dives deep into the world of **stochastic short-rate modelling**, implementing the **Cox-Ingersoll-Ross (CIR)** model from scratch, calibrating it against real noisy historical yield data, and extending it with a **Jump-Diffusion process** to handle sudden macroeconomic shocks.

The complete pipeline covers:

1. **Data Engineering** — Robust preprocessing with outlier detection and interpolation
2. **Base CIR Model** — Exact MLE calibration via non-central chi-square transition density
3. **Prediction Challenge** — Full yield curve reconstruction from the 3M rate alone
4. **Extension: CIR Jump-Diffusion** — Duffie-Pan-Singleton framework with compound Poisson jumps
5. **Critical Analysis** — Structural limitations, regime analysis, and failure modes

> **Key constraint:** During prediction, *only* the 3-Month yield is used as test-time input. All longer maturities are held-out actuals compared against model predictions.

---

## Core Problem Statement

> *How can a stochastic short-rate model be implemented, calibrated against noisy historical yield data, and extended to reconstruct an entire yield curve from a single observable input — and where do such models succeed or fail when confronted with real market dynamics?*

---

## Background & Theory

### The CIR SDE

The CIR model (Cox, Ingersoll & Ross, 1985) describes the evolution of the instantaneous short rate $r_t$ via:

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Speed of mean reversion | $\kappa > 0$ | How quickly rates revert to the long-run mean |
| Long-run mean | $\theta > 0$ | The equilibrium interest rate level |
| Volatility coefficient | $\sigma > 0$ | Amplitude of random fluctuations |
| Brownian motion | $W_t$ | Standard Wiener process |

The **Feller condition** $2\kappa\theta \geq \sigma^2$ ensures rates remain strictly positive.

### Zero-Coupon Bond Pricing

The closed-form price of a zero-coupon bond maturing at $T$:

$$P(t, T) = A(t, T)\,e^{-B(t,T)\,r_t}$$

where $A(t,T)$ and $B(t,T)$ are deterministic functions of the parameters and time-to-maturity $\tau = T - t$.

### Continuously Compounded Yield

$$y(t, \tau) = -\frac{\ln P(t,T)}{\tau} = \frac{B(t,T)\,r_t - \ln A(t,T)}{\tau}$$

---

## Repository Structure

```
├── Stochastic_Interest_Rate_Modelling_and_Prediction.ipynb   # Main notebook
├── data/
│   ├── train_data.csv          # Historical training yields
│   ├── test_data.csv           # Full test dataset (all maturities)
│   └── test_data_3M.csv        # Test dataset (3M yield only — model input)
├── README.md
└── Problem_statement.pdf       # Official problem statement
```

---

## Datasets

The dataset contains **daily zero-coupon bond yields** across **9 maturity tenors**:

| Column | Maturity |
|--------|----------|
| `ZC025YR` | 3 Months (0.25Y) |
| `ZC050YR` | 6 Months (0.50Y) |
| `ZC075YR` | 9 Months (0.75Y) |
| `ZC100YR` | 1 Year |
| `ZC200YR` | 2 Years |
| `ZC500YR` | 5 Years |
| `ZC1000YR` | 10 Years |
| `ZC2000YR` | 20 Years |
| `ZC3000YR` | 30 Years |

### Data Quality Notes
The raw data is **intentionally noisy**. It contains:
- Missing values (handled via interpolation and forward-filling)
- Outliers (detected and normalised using IQR-based methods)
- Formatting inconsistencies
- Non-trading day anomalies

> The underlying asset, geographical region, or market is undisclosed. The model relies purely on mathematical relationships within the data.

### Files

| File | Description |
|------|-------------|
| `train_data.csv` | Full training set with all 9 maturities |
| `test_data.csv` | Full test set with all 9 maturities (held-out actuals) |
| `test_data_3M.csv` | Test set with **only the 3M yield** — the sole model input during prediction |

---

## Workflow & Methodology

### A. Data Engineering & Preprocessing
- Load and validate all three dataset files
- Four-step automated cleaning pipeline:
  1. Parse and index dates
  2. Detect and handle outliers (IQR-based)
  3. Interpolate/forward-fill missing values
  4. Validate mathematical viability (positive yields)
- Exploratory analysis: yield distributions, correlation heatmaps, time-series plots

### B. Base CIR Model Calibration
- **Method:** Maximum Likelihood Estimation (MLE) using the exact non-central chi-square transition density of the CIR process
- **Justification:** MLE on the exact transition density is statistically efficient and avoids the discretisation bias inherent in OLS/GMM approaches
- Parameters estimated: $\kappa$, $\theta$, $\sigma$
- Feller condition verification and handling

### C. Yield Curve Prediction Challenge
- For each test day, **only the 3M yield** is ingested as a proxy for $r_t$
- Calibrated $A(\tau)$ and $B(\tau)$ functions are used to reconstruct yields at all other maturities
- Predicted vs. actual yield curves are visualised and evaluated

### D. Extension: CIR Jump-Diffusion
See [Model Extensions](#model-extensions) below.

### E. Critical Analysis
- Conditions under which the Feller condition breaks down
- Systematic over/underestimation by maturity bucket
- Regime analysis (normal vs. stress periods)
- Comparison of base CIR vs. extended model performance

---

## Model Extensions

### CIR Jump-Diffusion (Duffie-Pan-Singleton Framework)

The base CIR model is extended with a **compound Poisson jump process**:

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t + J_t\,dN_t$$

where:
- $N_t$ is a Poisson process with intensity $\lambda$ (jump arrival rate)
- $J_t$ are i.i.d. jump sizes drawn from an exponential distribution with mean $\mu_J$

**Motivation:** The base CIR model cannot capture sudden macroeconomic shocks (e.g., central bank policy surprises, crisis events). The Poisson jump component allows the model to produce discontinuous rate movements consistent with real market dynamics.

**Calibration:** Joint MLE over $(\kappa, \theta, \sigma, \lambda, \mu_J)$ using the approximate characteristic function method from Duffie, Pan & Singleton (2000).

---

## Evaluation Criteria

The submission is evaluated on two criteria:

| Criterion | Threshold |
|-----------|-----------|
| **Out-of-sample R²** on yield curve reconstruction from 3M rate | **> 0.85** |
| **Code quality** — Pythonic, modular, well-commented | Mandatory |

---

## Requirements

```
pandas
numpy
matplotlib
scipy
scikit-learn
```

Install via:
```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the three CSV files from the `data/` folder when prompted, or mount your Google Drive
3. Run all cells top-to-bottom: **Runtime → Run all**

### Option 2: Local Jupyter
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook Stochastic_Interest_Rate_Modelling_and_Prediction.ipynb
```

> **Note:** The notebook is self-contained. All data loading, preprocessing, model calibration, prediction, and evaluation are executed in sequence without requiring any external configuration.

---

## Key Results & Analysis

### Model Performance
- Base CIR achieves out-of-sample R² > 0.85 on yield curve reconstruction from the 3M rate alone
- Short-end maturities (6M–2Y) are well-fitted; long-end (20Y–30Y) show larger residuals due to the single-factor constraint
- The Jump-Diffusion extension improves fit during stress/high-volatility regimes

### Key Insights

**On Calibration:**
- MLE via non-central chi-square is more robust than OLS under the Feller-violating parameter regimes present in the data
- The estimated $\kappa$ implies [shock persistence analysis included in notebook]

**On the Feller Condition:**
- The condition $2\kappa\theta \geq \sigma^2$ is occasionally violated in high-volatility regimes
- When violated, a reflecting boundary condition is imposed in simulations to maintain positivity

**On Model Limitations:**
- Single-factor CIR constrains yield curve shapes to a limited family (monotone or humped)
- Cannot fit arbitrary initial term structures without time-dependent parameters (CIR++)
- Jump-Diffusion improves qualitative fit during shocks but introduces estimation challenges

---

## References

1. Cox, J.C., Ingersoll, J.E., & Ross, S.A. (1985). *A Theory of the Term Structure of Interest Rates.* Econometrica, 53(2), 385–407.
2. Duffie, D., Pan, J., & Singleton, K. (2000). *Transform Analysis and Asset Pricing for Affine Jump-Diffusions.* Econometrica, 68(6), 1343–1376.
3. Longstaff, F.A., & Schwartz, E.S. (1992). *Interest Rate Volatility and the Term Structure: A Two-Factor General Equilibrium Model.* Journal of Finance, 47(4), 1259–1282.
4. Brigo, D., & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice.* Springer Finance.

---

*Finance Club, IIT Roorkee — Open Projects 2026*
