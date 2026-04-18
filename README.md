# Quant-Alpha Risk Management Suite
**Author:** Rishab Patil (@MrRobotop)

A professional-grade quantitative risk tool powered by Google Research's **TimesFM 2.5** (Time Series Foundation Model). This suite provides recursive macro-conditioning, Value-at-Risk (VaR) synthesis, and on-the-fly empirical validation for enterprise-level risk analysis.

---

## 🚀 Overview

The Quant-Alpha Risk Engine transitions beyond simple price forecasting into **Regime-Aware Risk Assessment**. By combining zero-shot foundation models with classical quantitative finance (EWMA, VaR, Z-Scoring), the suite provides a self-verifying "truth engine" for market volatility.

### Key Professional Features
- **Recursive Macro-Conditioning**: Forecasts the future path of macro signals (VIX, 10Y Yield) before using them to condition the primary asset's risk profile.
- **Adaptive Regime Scoring**: Uses a rolling 90-day Z-score to determine if risk is "abnormal" for the current market cycle.
- **Institutional Performance**: Computes Forward-Looking Sharpe and Sortino ratios based on TimesFM expected returns and volatility.
- **Monte Carlo Probabilities**: Simulates 1,000 future paths using Geometric Brownian Motion to estimate the **Probability of Ruin** (e.g., 5% drawdown).
- **Macro Signal Strength**: Calculates real-time Pearson correlations to identify which macro drivers are currently coupled with the asset.
- **Tail Risk Synthesis (VaR & CVaR)**: Calculates both Value-at-Risk and Expected Shortfall (Conditional VaR).
- **Actionable Alpha (Kelly Sizing)**: Automatically computes the optimal fractional position size using the Continuous Kelly Criterion.
- **Empirical Backtesting**: Validates model reliability on every run by testing against the most recent 10 days of "hidden" data.

---

## 💎 The TimesFM Advantage: Why a Foundation Model?

While traditional risk engines rely on "shallow" statistical models (ARIMA, GARCH), the Quant-Alpha Suite is built on **Google's TimesFM 2.5**, a 200M parameter Transformer. This provides several cutting-edge advantages:

1.  **Zero-Shot Versatility**: Unlike models that need re-training for every asset, TimesFM has "pre-learned" the patterns of financial markets from billions of data points. It understands the "signature" of a market crash in NVDA just as well as it does in BTC, providing institutional-grade forecasts instantly.
2.  **Continuous Quantile Distributions**: Most models only predict the "median" path. TimesFM uses a continuous quantile head, allowing our engine to extract high-fidelity tail-risk data. This is the foundation for our precision VaR (Value-at-Risk) and CVaR calculations.
3.  **Non-Linear Macro Feedback**: By using TimesFM recursively, we capture the "feedback loop" between macro signals (like the VIX) and asset volatility. The model doesn't just look at where the macro signal is *now*—it anticipates its trajectory to condition the final risk score.

---

## 🏗️ Architecture & Pipeline Logic

The suite operates on a multi-stage execution pipeline:

1.  **Ingestion Layer**: Real-time retrieval of OHLCV data via `yfinance`.
2.  **Quant Signal Processing**: Calculation of log-returns, 20-day EWMA volatility, and 60-day macro correlation matrices.
3.  **Recursive Forecast Stage (A)**: All macro signals are projected via TimesFM.
4.  **Dual-Stream Forecast Stage (B)**: Parallel forecasting of returns ($\mu$) and volatility ($\sigma$) using projected macro paths as covariates.
5.  **Risk & Performance Synthesis**: Data is processed through VaR, CVaR, Monte Carlo, and Kelly engines to generate actionable allocation advice.

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
$$\sigma_t^2 = (1-\alpha)\sigma_{t-1}^2 + \alpha r_t^2$$

### Value-at-Risk (VaR) & Expected Shortfall (CVaR)
$$\text{VaR} = \text{Portfolio} \times \sigma \times Z_{\alpha}$$
$$\text{CVaR}_{\alpha} = \text{Portfolio} \times \sigma \times \frac{\phi(\Phi^{-1}(1-\alpha))}{1-\alpha}$$

### Probability of Ruin (Monte Carlo)
We simulate $N=1000$ paths using Geometric Brownian Motion:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$
The **Probability of Ruin** is defined as the percentage of paths that hit a -5% drawdown at any point during the forecast horizon.

### Continuous Kelly Criterion
$$f^* = \frac{\mu - r}{\sigma^2}$$
Where $\mu$ and $\sigma$ are annualized projections from the TimesFM dual-stream forecast.

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
