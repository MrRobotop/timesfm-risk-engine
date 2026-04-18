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
                "adaptive_z_score": float(risk_result["z_score"]),
                "var_exposure": float(risk_result["var_exposure"]),
                "model_reliability_pct": float(reliability) if reliability is not None else None
            },
            "status": risk_result["status"],
            "explanation": risk_result["explanation"]
        }
        print(json.dumps(payload, indent=2))
        return

    status = risk_result["status"]
    projected_vol = risk_result["projected_max_volatility"]
    z_score = risk_result["z_score"]
    var_exposure = risk_result["var_exposure"]
    explanation = risk_result["explanation"]
    
    table = Table(title="QUANT-ALPHA RISK MANAGEMENT SUITE", title_style="bold cyan", show_header=False, expand=True)
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
    table.add_row("[yellow]QUANTITATIVE METRICS", "")
    table.add_row("Latest Realized Vol (EWMA)", f"{float(latest_vol):.6f}")
    table.add_row("Projected Max Vol (90th Pct)", f"{float(projected_vol):.6f}")
    table.add_row("Adaptive Regime Z-Score", f"{float(z_score):.4f}")
    
    table.add_section()
    table.add_row("[yellow]EXPOSURE SYNTHESIS", "")
    table.add_row(f"Value-at-Risk ({int(args.confidence)}% VaR)", f"[bold red]${var_exposure:,.2f}[/bold red]")
    
    status_color = "red" if "DANGER" in status else "yellow" if "WARNING" in status else "green"
    
    # Diagnosis Panel
    diag_panel = Panel(
        Text(explanation, style="italic white"),
        title="[bold]RISK DIAGNOSIS[/bold]",
        border_style=status_color
    )
    
    status_panel = Panel(
        Text(status, style=f"bold {status_color}", justify="center"),
        title="[bold]SYSTEM STATUS[/bold]",
        border_style=status_color
    )
    
    console.print("\n")
    console.print(table)
    console.print(diag_panel)
    console.print(status_panel)
    console.print("\n")

def main():
    parser = argparse.ArgumentParser(description="TimesFM Quant-Alpha Risk Engine")
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
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--output-json", action="store_true")
    
    args = parser.parse_args()
    conf_decimal = args.confidence / 100.0
    
    # Resolve Tickers
    if args.preset != "custom":
        args.primary = args.primary or PRESETS[args.preset]["primary"]
        args.macros = args.macros or PRESETS[args.preset]["macros"]
    else:
        args.primary = args.primary or "SPY"
        args.macros = args.macros or "^VIX"
        
    macro_list = [m.strip() for m in args.macros.split(",")]
    
    with console.status(f"[bold green]Fetching data for {args.primary} + {args.macros}...", spinner="dots"):
        data_fetcher = MarketDataFetcher()
        primary_vol, macro_cov_dict = data_fetcher.fetch_multivariate_data(
            primary_ticker=args.primary, 
            macro_tickers=macro_list, 
            days=args.days + args.horizon,
            interval=args.interval
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
            bt_qf = forecaster.predict_with_macro(train_primary, train_macros, args.horizon)
            lower_90 = bt_qf[0, :, 1]
            upper_90 = bt_qf[0, :, 9]
            covered = np.sum((test_actual >= lower_90) & (test_actual <= upper_90))
            reliability = (covered / args.horizon) * 100

    # Final Forecast
    with console.status(f"[bold yellow]Executing Recursive Deep Forecaster...", spinner="dots"):
        if args.dynamic:
            quantile_forecast = forecaster.predict_dynamic_macro(primary_vol, macro_cov_dict, args.horizon)
        else:
            quantile_forecast = forecaster.predict_with_macro(primary_vol, macro_cov_dict, args.horizon)
    
    risk_result = assess_multivariate_risk(
        quantile_forecast=quantile_forecast, 
        historical_vol=primary_vol,
        risk_threshold=args.threshold,
        z_threshold=args.z_threshold,
        portfolio_value=args.portfolio,
        confidence_level=conf_decimal
    )
    
    print_dashboard(args, latest_realized_vol, risk_result, reliability)

    if args.export:
        output = {
            "risk_analysis": risk_result,
            "reliability": reliability,
            "metadata": vars(args)
        }
        with open(args.export, 'w') as f:
            json.dump(output, f, indent=2)
        console.print(f"[bold green]Report exported to {args.export}[/bold green]")

if __name__ == "__main__":
    main()
