import argparse
import sys
import json
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.data import MarketDataFetcher
from src.forecaster import RiskForecaster
from src.risk import assess_multivariate_risk

console = Console()

# Sector-Specific Risk Presets
PRESETS = {
    "custom": {"primary": "SPY", "macros": "^VIX"},
    "tech": {"primary": "NVDA", "macros": "QQQ,^VIX"},
    "crypto": {"primary": "BTC-USD", "macros": "ETH-USD,^VIX"},
    "macro": {"primary": "SPY", "macros": "^TNX,CL=F"},
    "safe-haven": {"primary": "GLD", "macros": "DX-Y.NYB,^VIX"}
}

def print_dashboard(args, latest_vol, risk_result, reliability=None):
    if args.output_json:
        payload = {
            "parameters": vars(args),
            "metrics": {
                "latest_realized_volatility": float(latest_vol),
                "projected_max_volatility": float(risk_result["projected_max_volatility"]),
                "expected_return_daily": float(risk_result["expected_return_daily"]),
                "adaptive_z_score": float(risk_result["z_score"]),
                "var_exposure": float(risk_result["var_exposure"]),
                "cvar_exposure": float(risk_result["cvar_exposure"]),
                "kelly_fraction": float(risk_result["kelly_fraction"]),
                "model_reliability_pct": float(reliability) if reliability is not None else None
            },
            "status": risk_result["status"],
            "explanation": risk_result["explanation"]
        }
        print(json.dumps(payload, indent=2))
        return

    status = risk_result["status"]
    projected_vol = risk_result["projected_max_volatility"]
    expected_ret = risk_result["expected_return_daily"]
    z_score = risk_result["z_score"]
    var_exp = risk_result["var_exposure"]
    cvar_exp = risk_result["cvar_exposure"]
    kelly_f = risk_result["kelly_fraction"]
    explanation = risk_result["explanation"]
    
    table = Table(title="QUANT-ALPHA PRO: RISK & ALLOCATION SUITE", title_style="bold cyan", show_header=False, expand=True)
    table.add_column("Key", style="bold white", justify="right")
    table.add_column("Value", style="bold magenta")
    
    table.add_section()
    table.add_row("[yellow]CONTEXT", "")
    table.add_row("Active Preset", args.preset.upper())
    table.add_row("Primary Asset", args.primary)
    table.add_row("Macro Covariates", args.macros)
    table.add_row("Portfolio Size", f"${args.portfolio:,.2f}")
    
    table.add_section()
    table.add_row("[yellow]ML RELIABILITY (BACKTEST)", "")
    if reliability is not None:
        rel_color = "green" if reliability > 80 else "yellow" if reliability > 60 else "red"
        table.add_row("Coverage Probability (90% CI)", f"[{rel_color}]{reliability:.1f}%[/{rel_color}]")
    else:
        table.add_row("Coverage Probability", "N/A (Backtest Skipped)")
        
    table.add_section()
    table.add_row("[yellow]QUANTITATIVE PROJECTIONS", "")
    table.add_row("Exp. Daily Return (Avg)", f"{expected_ret*100:+.4f}%")
    table.add_row("Latest Realized Vol (EWMA)", f"{float(latest_vol):.6f}")
    table.add_row("Projected Max Vol (90th Pct)", f"{float(projected_vol):.6f}")
    table.add_row("Adaptive Regime Z-Score", f"{float(z_score):.4f}")
    
    table.add_section()
    table.add_row("[yellow]TAIL RISK SYNTHESIS", "")
    table.add_row(f"Value-at-Risk ({int(args.confidence)}% VaR)", f"[bold red]${var_exp:,.2f}[/bold red]")
    table.add_row(f"Expected Shortfall ({int(args.confidence)}% CVaR)", f"[bold red]${cvar_exp:,.2f}[/bold red]")
    
    table.add_section()
    table.add_row("[yellow]ACTIONABLE ALPHA", "")
    kelly_color = "green" if kelly_f > 0.1 else "yellow" if kelly_f > 0 else "red"
    table.add_row("Optimal Kelly Allocation", f"[{kelly_color}]{kelly_f*100:.1f}% of Portfolio[/{kelly_color}]")
    
    status_color = "red" if "DANGER" in status else "yellow" if "WARNING" in status else "green"
    diag_panel = Panel(Text(explanation, style="italic white"), title="[bold]RISK DIAGNOSIS[/bold]", border_style=status_color)
    status_panel = Panel(Text(status, style=f"bold {status_color}", justify="center"), title="[bold]SYSTEM STATUS[/bold]", border_style=status_color)
    
    console.print("\n")
    console.print(table)
    console.print(diag_panel)
    console.print(status_panel)
    console.print("\n")

def main():
    parser = argparse.ArgumentParser(description="TimesFM Quant-Alpha Pro")
    parser.add_argument("--preset", type=str, default="custom", choices=list(PRESETS.keys()))
    parser.add_argument("--primary", type=str)
    parser.add_argument("--macros", type=str)
    parser.add_argument("--portfolio", type=float, default=1000000.0)
    parser.add_argument("--confidence", type=int, choices=[90, 95, 99], default=95)
    parser.add_argument("--dynamic", type=bool, default=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--export", type=str)
    parser.add_argument("--output-json", action="store_true")
    
    args = parser.parse_args()
    conf_decimal = args.confidence / 100.0
    
    if args.preset != "custom":
        args.primary = args.primary or PRESETS[args.preset]["primary"]
        args.macros = args.macros or PRESETS[args.preset]["macros"]
    else:
        args.primary = args.primary or "SPY"
        args.macros = args.macros or "^VIX"
        
    macro_list = [m.strip() for m in args.macros.split(",")]
    
    with console.status(f"[bold green]Fetching multi-signal data for {args.primary}...", spinner="dots"):
        data_fetcher = MarketDataFetcher()
        primary_vol, primary_returns, macro_cov_dict = data_fetcher.fetch_multivariate_data(
            primary_ticker=args.primary, macro_tickers=macro_list, days=args.days + args.horizon, interval=args.interval
        )
        latest_realized_vol = primary_vol[-1]
    
    with console.status("[bold blue]Initialising TimesFM Foundation Model...", spinner="point"):
        forecaster = RiskForecaster()

    # Empirical Backtest Logic
    reliability = None
    if len(primary_vol) > args.horizon * 2:
        with console.status(f"[bold magenta]Performing Empirical Backtest...", spinner="bouncingBar"):
            train_primary = primary_vol[:-args.horizon]
            test_actual = primary_vol[-args.horizon:]
            train_macros = {k: v[:-args.horizon] for k, v in macro_cov_dict.items()}
            _, bt_qf = forecaster.predict_with_macro(train_primary, train_macros, args.horizon)
            covered = np.sum((test_actual >= bt_qf[0, :, 1]) & (test_actual <= bt_qf[0, :, 9]))
            reliability = (covered / args.horizon) * 100

    # Final Forecast: Dual Stream (Volatility + Returns)
    with console.status(f"[bold yellow]Executing Recursive Dual-Stream Forecaster...", spinner="dots"):
        if args.dynamic:
            vol_p, vol_q = forecaster.predict_dynamic_macro(primary_vol, macro_cov_dict, args.horizon)
            ret_p, _ = forecaster.predict_dynamic_macro(primary_returns, macro_cov_dict, args.horizon)
        else:
            vol_p, vol_q = forecaster.predict_with_macro(primary_vol, macro_cov_dict, args.horizon)
            ret_p, _ = forecaster.predict_with_macro(primary_returns, macro_cov_dict, args.horizon)
    
    # Calculate average expected daily return from point forecast
    avg_expected_return = np.mean(ret_p)
    
    risk_result = assess_multivariate_risk(
        quantile_forecast=vol_q, 
        historical_vol=primary_vol,
        risk_threshold=args.threshold,
        z_threshold=args.z_threshold,
        portfolio_value=args.portfolio,
        confidence_level=conf_decimal,
        expected_return_daily=avg_expected_return
    )
    
    print_dashboard(args, latest_realized_vol, risk_result, reliability)

    if args.export:
        output = {"risk_analysis": risk_result, "reliability": reliability, "metadata": vars(args)}
        with open(args.export, 'w') as f: json.dump(output, f, indent=2)
        console.print(f"[bold green]Report exported to {args.export}[/bold green]")

if __name__ == "__main__": main()
