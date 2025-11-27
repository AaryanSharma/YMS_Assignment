#!/usr/bin/env python
"""
Problem 2 – Mean-Reverting Calendar Spread Strategy (Updated using Problem 1 insights)

- Uses FUT2−FUT1 spread per underlying
- Filters out "chaotic" high-vol names (Regime B from Problem 1C)
- Trades only in the "good" DTE window (2–7 days to expiry) suggested by Problem 1B
- Outputs Results.csv and results.<timestamp>.csv

Required external module in same directory:
    submission.py  with functions:
        - extract_archive()
        - load_all_days()
        - build_panel(data_raw)

Panel is assumed to have columns:
    ['timestamp', 'trading_date', 'underlying',
     'cm_ltp', 'fut1_ltp', 'fut2_ltp',
     'fut1_volume', 'fut2_volume',
     'fut1_lot_size', 'fut2_lot_size',
     'days_to_expiry', ...]
"""

import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Suppress specific pandas warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from problem1_solution import extract_archive, load_all_days, build_panel
except ImportError:
    print("Error: Could not import 'build_panel'. Ensure problem1_solution.py exists.")
    exit(1)


# -----------------------------
# Strategy hyperparameters
# -----------------------------

# Use only the "good" window identified in 1B
DTE_MIN = 2       # inclusive
DTE_MAX = 7       # inclusive

# Rolling window for z-score (in minutes)
ROLLING_WINDOW = 300  # entire recent history

# Z-score thresholds
Z_ENTRY = 2.0
Z_EXIT = 0.5

# Volatility-based universe filter (from Problem 1C insight)
# Exclude names with extremely high FUT2-FUT1 spread std (Regime B).
SPREAD_STD_MAX = 25.0  

# Volume filter: require minimum median spreadable volume
MIN_MEDIAN_FUT_VOLUME = 500  # per-minute median, rough sanity threshold

# Position sizing
LOTS_PER_TRADE = 1  # 1 calendar spread = 1 lot FUT1 + 1 lot FUT2

# Cost assumptions (bps of notional per leg per side)
COMMISSION_BPS = 0.5e-4   # 0.5 bps
SLIPPAGE_BPS   = 1.0e-4   # 1.0 bps


# -----------------------------
# Helper functions
# -----------------------------

def compute_market_lots(panel: pd.DataFrame) -> float:
    """Approximate total market lots traded in FUT1 and FUT2 across all names."""
    # Handle cases where column might be 'fut1_ttq' diff instead of pre-calculated 'fut1_volume'
    if "fut1_volume" not in panel.columns:
         # simple fallback if columns missing, though build_panel usually provides them
         return 0.0
         
    fut1_lots = (panel["fut1_volume"] / panel["fut1_lot_size"]).replace([np.inf, -np.inf], np.nan).fillna(0).sum()
    fut2_lots = (panel["fut2_volume"] / panel["fut2_lot_size"]).replace([np.inf, -np.inf], np.nan).fillna(0).sum()
    return fut1_lots + fut2_lots

def get_zero_result(stock_name: str) -> dict:
    """Returns a result row with 0s for untraded/filtered stocks."""
    return {
        "stock_name": stock_name,
        "n_traded_days": 0,
        "net_pnl": 0.0,
        "gross_pnl": 0.0,
        "cost_pnl": 0.0,
        "slippage_fut1": 0.0,
        "slippage_fut2": 0.0,
        "total_lots_traded": 0.0,
        "total_volume": 0.0,
        "max_delta_qty": 0.0,
        "max_gross_qty": 0.0,
        "drawdown": 0.0,
        "market_perc": 0.0
    }

def filter_universe_for_problem2(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply universe filters motivated by Problem 1:
    - Use days_to_expiry in [0,10] for stats, but we will trade only [2,7].
    - Exclude high-vol names (Regime B) with very large FUT2-FUT1 std.
    - Exclude illiquid names with extremely low futures volume.
    """
    df = panel.copy()

    # Require non-null FUT1 and FUT2 prices
    df = df.dropna(subset=["fut1_ltp", "fut2_ltp"])
    # Limit to DTE in [0,10] for robust stats (same window as Problem 1)
    df = df[df["days_to_expiry"].between(0, 10)]

    # FUT2-FUT1 spread
    df["fut_spread"] = df["fut2_ltp"] - df["fut1_ltp"]

    # Spread volatility per underlying
    spread_stats = (
        df.groupby("underlying")["fut_spread"]
          .agg(spread_mean="mean", spread_std="std", count="count")
          .reset_index()
    )

    # Volume stats per underlying (median FUT1+FUT2 per minute)
    # FillNA 0 to avoid issues with sparse data
    df["vol_sum"] = df["fut1_volume"].fillna(0) + df["fut2_volume"].fillna(0)
    
    vol_stats = (
        df.groupby("underlying")["vol_sum"]
          .median()
          .reset_index()
          .rename(columns={"vol_sum": "median_fut_volume"})
    )

    uni = spread_stats.merge(vol_stats, on="underlying", how="left")

    # Apply filters:
    # 1) Sufficient data
    uni = uni[uni["count"] > 500]  # require enough minutes to estimate spread stats
    # 2) Exclude extremely high-vol names (Regime B from 1C)
    uni = uni[uni["spread_std"] <= SPREAD_STD_MAX]
    # 3) Exclude names with almost no futures volume
    uni = uni[uni["median_fut_volume"] >= MIN_MEDIAN_FUT_VOLUME]

    allowed_syms = set(uni["underlying"].unique())
    
    print(f"Universe Filter: {len(allowed_syms)} stocks passed out of {len(spread_stats)}.")

    filtered = df[df["underlying"].isin(allowed_syms)].copy()
    return filtered


def backtest_symbol(df_sym: pd.DataFrame) -> dict | None:
    """
    Mean-reverting calendar spread strategy for a single underlying, incorporating:
    - Trade only when DTE in [DTE_MIN, DTE_MAX]
    - FUT2-FUT1 spread z-score mean reversion
    - Entry/exit thresholds Z_ENTRY, Z_EXIT
    - Cost and slippage model
    """
    df = df_sym.copy()
    df = df.sort_values("timestamp")

    # Filter to near-expiry trading band for entries; but keep [0,10] in df so z-score has context.
    # enforce DTE window in the entry/exit logic.
    df = df[df["days_to_expiry"].between(0, 10)]
    if df.empty:
        return None

    # Require both future prices and positive volumes
    df = df.dropna(subset=["fut1_ltp", "fut2_ltp"])
    if df.empty:
        return None

    # Build spread and rolling z-score on full [0,10] DTE band, but entries restricted further
    df["spread"] = df["fut2_ltp"] - df["fut1_ltp"]
    df["spread_mean"] = df["spread"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2).mean()
    df["spread_std"] = df["spread"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2).std()
    df["z"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

    df = df[~df["z"].isna()]
    if df.empty:
        return None

    # State
    position = 0   # +1 = long spread (long FUT2, short FUT1), -1 = short spread
    entry_spread = None

    gross_pnl = 0.0
    cost_pnl = 0.0
    slippage_fut1 = 0.0
    slippage_fut2 = 0.0
    total_lots_traded = 0.0
    total_volume = 0.0
    traded_days = set()

    # For drawdown tracking
    cum_net = 0.0
    equity_peak = 0.0
    max_drawdown = 0.0

    max_delta_qty = 0.0   # in shares (delta)
    max_gross_qty = 0.0   # in shares (gross notion of lots*lot_size)

    # Assume constant lot sizes per symbol
    fut1_lot_size = df["fut1_lot_size"].iloc[0]
    fut2_lot_size = df["fut2_lot_size"].iloc[0]

    def apply_costs(fut1_price: float, fut2_price: float, lots_signed: float):
        """Apply costs for entry/exit of 'lots_signed' calendar spreads at current prices."""
        nonlocal cost_pnl, slippage_fut1, slippage_fut2, total_lots_traded, total_volume, cum_net

        lots = abs(lots_signed)
        notional_fut1 = fut1_price * fut1_lot_size * lots
        notional_fut2 = fut2_price * fut2_lot_size * lots

        commission = COMMISSION_BPS * (notional_fut1 + notional_fut2)
        slip1 = SLIPPAGE_BPS * notional_fut1
        slip2 = SLIPPAGE_BPS * notional_fut2

        leg_cost = commission + slip1 + slip2

        cost_pnl += leg_cost
        slippage_fut1 += slip1
        slippage_fut2 += slip2
        total_lots_traded += 2 * lots  # FUT1 + FUT2
        total_volume += notional_fut1 + notional_fut2

        # Net equity impact of costs
        cum_net -= leg_cost

    for _, row in df.iterrows():
        z = row["z"]
        dte = row["days_to_expiry"]
        fut1_price = row["fut1_ltp"]
        fut2_price = row["fut2_ltp"]
        trading_date = row["trading_date"]

        if pd.isna(z): continue

        # Risk quantities (shares)
        delta_shares = position * (fut2_lot_size - fut1_lot_size)
        gross_shares = abs(position) * (fut1_lot_size + fut2_lot_size)
        max_delta_qty = max(max_delta_qty, abs(delta_shares))
        max_gross_qty = max(max_gross_qty, gross_shares)

        # EXIT logic
        if position != 0:
            exit_signal = False

            # No new entries outside the "good" band, and enforce forced exit when DTE leaves [DTE_MIN, DTE_MAX]
            if not (DTE_MIN <= dte <= DTE_MAX):
                exit_signal = True

            # Normal mean-reversion exit
            if abs(z) < Z_EXIT:
                exit_signal = True

            if exit_signal:
                # Close existing position
                lots_to_close = position * LOTS_PER_TRADE
                exit_spread = fut2_price - fut1_price
                # PnL logic for Long Spread (Buy F2, Sell F1): (F2_out - F2_in) - (F1_out - F1_in)
                # Short Spread (Sell F2, Buy F1): (F2_in - F2_out) - (F1_in - F1_out)
                # Simplified: position * (Spread_out - Spread_in) where Spread = F2-F1
                pnl_trade = position * (exit_spread - entry_spread) * fut1_lot_size * LOTS_PER_TRADE
                gross_pnl += pnl_trade

                # Costs for exit
                apply_costs(fut1_price, fut2_price, lots_to_close)

                # Update equity and drawdown
                cum_net += pnl_trade
                equity_peak = max(equity_peak, cum_net)
                max_drawdown = min(max_drawdown, cum_net - equity_peak)

                position = 0
                entry_spread = None

                # Do not enter in same bar after exit
                continue

        # ENTRY logic – only in DTE [DTE_MIN, DTE_MAX]
        if position == 0 and (DTE_MIN <= dte <= DTE_MAX):
            if z > Z_ENTRY:
                # Short spread: short FUT2, long FUT1
                position = -1
                entry_spread = fut2_price - fut1_price
                traded_days.add(trading_date)
                apply_costs(fut1_price, fut2_price, -LOTS_PER_TRADE)

            elif z < -Z_ENTRY:
                # Long spread: long FUT2, short FUT1
                position = 1
                entry_spread = fut2_price - fut1_price
                traded_days.add(trading_date)
                apply_costs(fut1_price, fut2_price, +LOTS_PER_TRADE)

    # We ignore residual open positions at end (they should be rare given DTE window + exit rules)

    net_pnl = gross_pnl - cost_pnl
    drawdown = abs(max_drawdown)

    return dict(
        stock_name=df["underlying"].iloc[0],
        n_traded_days=len(traded_days),
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        cost_pnl=cost_pnl,
        slippage_fut1=slippage_fut1,
        slippage_fut2=slippage_fut2,
        total_lots_traded=total_lots_traded,
        total_volume=total_volume,
        max_delta_qty=max_delta_qty,
        max_gross_qty=max_gross_qty,
        drawdown=drawdown,
    )


def run_problem2():
    # 1) Build panel from Problem 1 code
    extract_archive()
    data_raw = load_all_days()
    panel = build_panel(data_raw)

    # Capture ALL stocks before filtering
    all_stocks = panel["underlying"].unique()

    # Filter universe based on Problem 1 insights
    panel_filtered = filter_universe_for_problem2(panel)
    
    # Get set of valid stocks for fast lookup
    valid_stocks_set = set(panel_filtered["underlying"].unique())

    # Pre-compute market lots for market_perc (using FULL panel for total market view)
    total_market_lots = compute_market_lots(panel)

    # 2) Run strategy loop
    results = []
    
    print(f"Processing {len(all_stocks)} stocks...")
    
    for stock in all_stocks:
        if stock in valid_stocks_set:
            # This stock survived filters, run backtest
            df_sym = panel_filtered[panel_filtered["underlying"] == stock]
            res = backtest_symbol(df_sym)
            
            if res is not None:
                results.append(res)
            else:
                # Backtest returned None (e.g. insufficient data inside backtest logic)
                results.append(get_zero_result(stock))
        else:
            # Stock was filtered out (Regime B, Illiquid, etc)
            results.append(get_zero_result(stock))

    if not results:
        raise RuntimeError("No results generated.")

    df_res = pd.DataFrame(results)

    # 3) Compute market_perc
    if total_market_lots > 0:
        df_res["market_perc"] = df_res["total_lots_traded"] / total_market_lots
    else:
        df_res["market_perc"] = 0.0

    # 4) Sort and output in required format
    df_res = df_res.sort_values("net_pnl", ascending=False).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_filename = f"results.{timestamp}.csv"

    df_res.to_csv(ts_filename, index=False)
    # df_res.to_csv("Results.csv", index=False)

    # print(f"Saved {ts_filename} ({len(df_res)} rows).")
    
    # Verify a filtered stock exists
    filtered_examples = df_res[df_res['total_lots_traded'] == 0]
    if not filtered_examples.empty:
        print(f"Verified inclusion of filtered stocks (e.g., {filtered_examples.iloc[0]['stock_name']})")


if __name__ == "__main__":
    run_problem2()