---
name: quant-alpha-risk-engine
description: >
  Professional-grade quantitative risk management suite powered by Google's TimesFM 2.5.
  Features recursive macro-conditioning, Value-at-Risk (VaR) synthesis, and 
  empirical backtest validation. Supports sector presets (Tech, Crypto, Macro) 
  and dynamic covariate path forecasting.
license: Apache-2.0
metadata:
  author: Clayton Young (@borealBytes)
  version: "2.0.0"
---

# Quant-Alpha Risk Management Suite

## Overview

The Quant-Alpha Risk Management Suite is a professional-grade quantitative tool powered by 
Google Research's **TimesFM 2.5**. It transitions beyond simple forecasting by 
incorporating multi-stage macro-conditioning, portfolio exposure synthesis, and 
on-the-fly empirical validation.

## Key Features

### 1. Recursive Macro-Conditioning (Dynamic XReg)
Unlike standard models that assume static macro environments, this suite uses a 
**Two-Stage Forecast**:
1.  **Stage A**: Forecasts the future expected path of all macro signals (VIX, 10Y Yield, etc).
2.  **Stage B**: Uses these projected paths as dynamic covariates for the primary asset.

### 2. Value-at-Risk (VaR) Synthesis
Translates abstract volatility percentages into actionable dollar-value risk.
- **95% VaR**: The maximum expected 1-day loss on your portfolio with 95% confidence.
- Formula: `Portfolio Value * Projected Volatility * 1.645`.

### 3. Empirical Calibration (Backtesting)
Automatically validates the model's reliability for every ticker by performing a 
rolling backtest on the most recent historical period.
- **Coverage Probability**: Percentage of actual price movements captured within the model's 90% Confidence Interval.

## Available Risk Presets

| Preset | Primary Asset | Macro Covariates | Use Case |
| ------ | ------------- | ---------------- | -------- |
| `tech` | NVDA | QQQ, ^VIX | High-growth semi/software exposure |
| `crypto`| BTC-USD | ETH-USD, ^VIX | Extreme volatility digital assets |
| `macro` | SPY | ^TNX, CL=F | Global equity vs Rates & Energy |
| `safe-haven`| GLD | DX-Y.NYB, ^VIX | Gold vs Dollar strength |

## 🎯 Usage

### Professional Dashboard
```bash
python main.py --preset tech --portfolio 1000000 --horizon 10
```

### Advanced Manual Selection
```bash
python main.py --primary TSLA --macros QQQ,^VIX,TLT --days 500 --dynamic True
```

## 📊 Understanding the Dashboard

| Metric | Interpretation |
| ------ | -------------- |
| **Coverage Probability** | Reliability score. >80% indicates the model handles this asset's current regime well. |
| **Adaptive Z-Score** | How "unusual" the forecasted volatility is relative to the last 90 days. |
| **95% VaR** | The dollar amount "at risk" over the next 24 hours. |

## ⚠️ Preflight: System Requirements Check

TimesFM 2.5 uses ~800 MB on disk and ~1.5 GB in RAM.

```bash
# Recommended hardware
# RAM: ≥ 4 GB
# VRAM: ≥ 2 GB (Optional, CUDA/MPS supported)
```
