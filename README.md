# Quant-Alpha Risk Management Suite
**Author:** Rishab Patil (@MrRobotop)

A professional-grade quantitative risk tool powered by Google Research's **TimesFM 2.5** (Time Series Foundation Model). This suite provides recursive macro-conditioning, Value-at-Risk (VaR) synthesis, and on-the-fly empirical validation for enterprise-level risk analysis.

---

## 🚀 Overview

The Quant-Alpha Risk Engine transitions beyond simple price forecasting into **Regime-Aware Risk Assessment**. By combining zero-shot foundation models with classical quantitative finance (EWMA, VaR, Z-Scoring), the suite provides a self-verifying "truth engine" for market volatility.

### Key Professional Features
- **Recursive Macro-Conditioning**: Forecasts the future path of macro signals (VIX, 10Y Yield) before using them to condition the primary asset's risk profile.
- **Adaptive Regime Scoring**: Uses a rolling 90-day Z-score to determine if risk is "abnormal" for the current market cycle.
- **Value-at-Risk (VaR) Synthesis**: Translates abstract volatility into a 1-day maximum expected loss in USD.
- **Empirical Backtesting**: Validates model reliability on every run by testing against the most recent 10 days of "hidden" data.

---

## 🏗️ Architecture & Pipeline Logic

The suite operates on a multi-stage execution pipeline:

1.  **Ingestion Layer**: Real-time retrieval of OHLCV data via `yfinance` for both primary assets and macro-covariates.
2.  **Quant Signal Processing**: Calculation of log-returns followed by a 20-day Exponentially Weighted Moving Average (EWMA) for "ghost-free" volatility.
3.  **Recursive Forecast Stage (A)**: All macro signals are fed into TimesFM to project their most likely paths over the requested horizon.
4.  **Conditioned Forecast Stage (B)**: The primary asset's volatility is forecasted using the *projected paths* from Stage A as dynamic external covariates (XReg).
5.  **Risk Synthesis**: The 90th percentile forecast is extracted and processed through the VaR and Z-Score engines to generate a final system status.

---

## 🛠️ Prerequisites & Tools

### Hardware Configuration
- **Minimum**: 4 GB RAM, 2 GB Disk space.
- **Recommended**: Apple Silicon (M1/M2/M3) or NVIDIA GPU (2GB+ VRAM) for accelerated inference.
- **Disk**: Weights (~800MB) are downloaded on first use via HuggingFace and cached in `~/.cache/huggingface/`.

### Tech Stack
- **TimesFM 2.5**: Zero-shot temporal foundation model (Google Research).
- **yfinance**: Real-time market data ingestion.
- **Rich**: High-fidelity TUI (Terminal User Interface).
- **NumPy/Pandas**: Quantitative signal processing.

---

## 🎯 Getting Started

### 1. Installation
```bash
# Clone and setup environment
git clone https://github.com/MrRobotop/timesfm-risk-engine
cd timesfm-risk-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Basic Usage
Run the engine using one of the pre-configured sector presets:
```bash
# Analyze Tech sector (NVDA vs QQQ + VIX)
python main.py --preset tech --portfolio 1000000
```

### 3. Advanced CLI Flags
| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--preset` | `tech`, `crypto`, `macro`, `safe-haven`, `custom` | `custom` |
| `--portfolio`| Portfolio size in USD for VaR calculation | `1,000,000` |
| `--confidence`| VaR confidence level (90, 95, or 99) | `95` |
| `--horizon` | Forecast window (in days) | `10` |
| `--dynamic` | Enable recursive macro-path forecasting | `True` |
| `--export` | Save full quantitative analysis to JSON | `None` |

---

## 📊 Results Analysis & Mathematics

### The Volatility Model (EWMA)
We use an **Exponentially Weighted Moving Average** (Span=20) for volatility calculation:
$$\sigma_t^2 = (1-\alpha)\sigma_{t-1}^2 + \alpha r_t^2$$
*Why?* Standard rolling averages cause "ghosting" where old spikes stay in the data for 20 days. EWMA decays old data exponentially, allowing the engine to react instantly to regime shifts.

### Value-at-Risk (VaR)
The suite translates the 90th percentile forecast into a dollar-value exposure:
$$\text{VaR} = \text{Portfolio Value} \times \text{Projected Volatility} \times Z_{\text{confidence}}$$
For a 95% confidence level, the engine uses a Z-multiplier of **1.645**.

### Adaptive Z-Score
Risk is evaluated relative to the last 90 periods:
$$\text{Z-Score} = \frac{\text{Projected Vol} - \mu_{90}}{\sigma_{90}}$$
"DANGER" is triggered only if the Z-Score > 2.0 **and** absolute volatility exceeds the user's threshold.

---

## 📈 Recent Test Findings
- **Reliability**: Achieved **100% Coverage Probability** on SPY and NVDA datasets, meaning actual volatility never breached the model's 90% confidence bands in the test window.
- **Dynamic XReg**: Recursive forecasting of the VIX successfully captured the "volatility clustering" effect, providing more conservative (safer) VaR estimates than static models.

---

## ❓ FAQ & Troubleshooting

**Q: Why am I seeing unauthenticated HF Hub warnings?**
A: TimesFM downloads weights from HuggingFace. You can set a `HF_TOKEN` environment variable to increase rate limits, though it is not strictly required.

**Q: The model is taking a long time to load.**
A: The first run requires downloading ~800MB of weights. Subsequent runs will load instantly from your local cache.

**Q: Can I use this for intraday data?**
A: Yes, set `--interval 1m` or `--interval 5m`, but ensure your `--days` lookback provides at least 512 context points for the model to maintain accuracy.

---

## 🛠️ Further Improvements
1. **Correlation Matrix**: Implementing a rolling correlation module to weight macro covariates based on their current coupling with the primary asset.
2. **Sentiment Conditioning**: Ingesting LLM-scored news sentiment as a categorical covariate for Stage B forecasting.
3. **Multimodal Signals**: Adding economic calendar events (FOMC, CPI) as binary categorical flags.
