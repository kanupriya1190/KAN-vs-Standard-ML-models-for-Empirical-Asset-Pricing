# Empirical Asset Pricing with Kolmogorov-Arnold Networks

**Comparative Analysis of KANs and Standard ML Models for Cross-Sectional Return Forecasting**

Master's Capstone · MS Quantitative Economics · UCLA · 2025
Advisor: Denis Chetverikov

---

## Overview

This project benchmarks **Kolmogorov-Arnold Networks (KANs)** against established machine learning models for predicting monthly cross-sectional stock returns among the **Magnificent 7** U.S. mega-cap equities. Motivated by the empirical asset pricing framework of [Gu, Kelly & Xiu (2020)](https://doi.org/10.1093/rfs/hhab083), it evaluates models on both **statistical accuracy** (out-of-sample R²) and **economic value** (long-short portfolio Sharpe ratios and maximum drawdown).

**Key finding:** All models produce negative out-of-sample R² in this concentrated, high-efficiency universe — but GBRT still extracts a modest positive Sharpe (0.74) from nonlinear patterns, while neural architectures (DNN, RNN, KAN) and linear regression fail to generate persistent portfolio value.

---

## Results

| Model | OOS R² | Sharpe Ratio | Max Drawdown |
|-------|--------|-------------|--------------|
| **KAN** | −4.11 | 0.02 | −48.91% |
| **DNN** | −7.95 | −0.90 | −89.34% |
| **RNN** | −0.68 | −0.92 | −82.71% |
| **GBRT** | −1.50 | **0.74** | **−43.17%** |
| **OLS** | −0.32 | −0.63 | −85.39% |

GBRT is the only model that delivers a positive Sharpe ratio and avoids catastrophic drawdowns, suggesting tree-based nonlinearities retain some exploitable signal even in an extremely efficient market segment.

---

## Universe & Data

**Stocks:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
**Period:** January 2010 – December 2024 (monthly frequency)

### Data Sources

| Source | What it provides |
|--------|-----------------|
| Yahoo Finance (`yfinance`) | Monthly adjusted close prices, sector metadata |
| FRED (`fredapi`) | 3-month T-Bill (DTB3), Term Spread (T10Y2Y), VIX (VIXCLS) |

### Engineered Features

- **12-month Momentum** — cumulative return over prior 12 months (excluding most recent month)
- **3-month Volatility** — rolling standard deviation of monthly returns
- **Short-term Reversal** — negative of previous month's return
- **Past Return** — lagged monthly return
- **Macro variables** — T-Bill rate, term spread, VIX
- **Interaction terms** — momentum × macro, volatility × macro
- **Sector dummies** — one-hot encoded from Yahoo Finance metadata

All features are standardized before model training.

---

## Models

### Kolmogorov-Arnold Network (KAN)

Architecture grounded in the Kolmogorov-Arnold representation theorem. Replaces standard linear inter-layer transformations with learnable univariate nonlinear functions (spline-based). Configured as `[n_features, 64, 32, 1]` with grid resolution 5 and spline order 3. Trained for 50 epochs with Adam optimizer (lr=1e-3).

### Deep Neural Network (DNN)

Feed-forward network with ReLU activations. Architecture: `[n_features → 64 → 32 → 1]`. Trained for 50 epochs, Adam optimizer, MSE loss.

### Recurrent Neural Network (RNN)

Simple RNN with hidden dimension 32. Each sample treated as a length-1 sequence (cross-sectional setup, not sequential). Trained for 50 epochs.

### Gradient-Boosted Regression Trees (GBRT)

Scikit-learn `GradientBoostingRegressor` with 200 estimators, max depth 3, learning rate 0.05, subsample 0.8.

### Ordinary Least Squares (OLS)

Standard linear regression baseline using `LinearRegression` from scikit-learn.

---

## Evaluation Methodology

### Statistical: Out-of-Sample R²

Measures whether model predictions improve upon the naive training-set mean forecast. Negative values indicate worse-than-mean performance.

### Economic: Long-Short Portfolio Backtesting

Each month, stocks are ranked by predicted return and sorted into deciles. A long-short portfolio goes long the top decile and short the bottom decile. Performance is measured via:

- **Annualized Sharpe Ratio** — average excess return per unit of risk
- **Maximum Drawdown** — largest peak-to-trough decline in cumulative portfolio value

This follows the portfolio-sorting methodology of Gu, Kelly & Xiu (2020).

---

## Repository Structure

```
├── capstone_.ipynb          # Full implementation notebook
├── Kanupriya_Capstone_2025.pdf  # Written thesis/paper
├── kelly_style_dataset.csv  # Generated feature dataset (after running cells 0-9)
└── README.md
```

### Notebook Walkthrough

| Cells | Stage |
|-------|-------|
| 0–2 | Data download (Yahoo Finance + FRED) |
| 3–6 | Feature engineering (momentum, volatility, reversal, macro, interactions, sector dummies) |
| 7–9 | Data cleaning, export to CSV |
| 10 | Train/test split, scaling, PyTorch data loaders |
| 11 | KAN training and evaluation |
| 12 | DNN training and evaluation |
| 13 | RNN training and evaluation |
| 14 | GBRT training and evaluation |
| 15 | OLS baseline |
| 16–17 | Predictions assembly into test dataframe |
| 18 | Decile portfolio assignment |
| 19 | Long-short return computation + Sharpe ratios |
| 20–22 | Individual model cumulative return plots |
| 23 | Maximum drawdown calculation |
| 24 | Combined cumulative return plot (all models) |

---

## Setup

### Requirements

```
python >= 3.9
yfinance
pandas
numpy
scikit-learn
torch
fredapi
pykan          # pip install git+https://github.com/KindXiaoming/pykan.git
matplotlib
tqdm
```

### API Keys

A FRED API key is required for macroeconomic data. Get one free at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).

### Running

```bash
# 1. Install dependencies
pip install yfinance pandas numpy scikit-learn torch fredapi matplotlib tqdm
pip install git+https://github.com/KindXiaoming/pykan.git

# 2. Open and run the notebook end-to-end
jupyter notebook capstone_.ipynb
```

Run cells sequentially — later cells depend on variables from earlier ones.

---

## Key Takeaways

1. **Market efficiency matters.** In a narrow universe of 7 mega-cap stocks with deep analyst coverage, all models fail to beat the mean forecast statistically — validating that ML's asset pricing power depends on cross-sectional breadth and return dispersion.

2. **Economic evaluation is essential.** GBRT achieves a 0.74 Sharpe despite negative R², showing that statistical fit and portfolio value can diverge — a core insight from Gu, Kelly & Xiu (2020).

3. **KANs show promise but not dominance.** KAN avoids the catastrophic drawdowns of DNN/RNN and achieves a near-zero Sharpe (0.02), suggesting the architecture has some capacity to avoid overfitting, but does not unlock new alpha in this setting.

4. **Tree-based methods remain robust.** GBRT's ability to capture nonlinear interactions without overfitting to noise makes it the most reliable model even in this adversarial environment.

---

## References

- Gu, S., Kelly, B. T., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Liu, Z., Liu, S., & Lin, K. (2023). Kolmogorov–Arnold Networks. *arXiv:2305.17143*.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Chen, T., & Pelger, M. (2021). Deep Learning in Asset Pricing. *Review of Financial Studies*, 34(11), 5149–5202.

---

## Author

**Kanupriya Parashar**
MS Quantitative Economics, UCLA · [GitHub](https://github.com/kanupriya1190) · [LinkedIn](https://linkedin.com/in/kanupriya-parashar)
