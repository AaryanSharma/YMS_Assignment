#!/usr/bin/env python

import os
import glob
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
import tarfile

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

ARCHIVE_TAR = "customdata_new.tar.gz"   
EXTRACT_DIR = "customdata_new"            # where CSVs will be extracted
OUTPUT_DIR = "outputs_problem1"           # where plots will be saved

NEAR_DTE_MAX = 10  # "near expiry" window: days_to_expiry <= 10

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

FUT_PATTERN = re.compile(r"^(?P<underlying>[A-Z0-9]+)(?P<yy>\d{2})(?P<mon>[A-Z]{3})FUT$")


# ----------------------------------------------------------------------
# UTILS: extraction, date helpers
# ----------------------------------------------------------------------

def extract_archive():
    if os.path.exists(EXTRACT_DIR):
        return
    if os.path.exists(ARCHIVE_TAR):
        print(f"Extracting {ARCHIVE_TAR} to {EXTRACT_DIR} ...")
        with tarfile.open(ARCHIVE_TAR, "r:gz") as tf:
            tf.extractall(EXTRACT_DIR)
    else:
        raise FileNotFoundError(
            f"{ARCHIVE_TAR} not found. "
            f"Place your archive in the current directory."
        )


def last_thursday_of_month(year: int, month: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - 3) % 7  # Thursday = 3
    return last_day - timedelta(days=offset)


def parse_fut_name(name: str):
    """
    Parse futures name like GAIL25FEBFUT -> (underlying, expiry_date (date))
    """
    m = FUT_PATTERN.match(name)
    if m is None:
        return None, None

    underlying = m.group("underlying")
    yy = int(m.group("yy"))
    mon_code = m.group("mon")
    month = MONTH_MAP.get(mon_code)
    if month is None:
        return None, None

    year = 2000 + yy
    expiry = last_thursday_of_month(year, month)
    return underlying, expiry


# ----------------------------------------------------------------------
# LOAD AND BUILD RAW DATAFRAME
# ----------------------------------------------------------------------

def load_all_days():
    all_dfs = []

    # Recursive search for ALL csv files under EXTRACT_DIR
    csv_paths = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)

    if not csv_paths:
        raise RuntimeError(
            f"No CSV files found under {EXTRACT_DIR}. "
            f"Check extraction structure or file extensions."
        )

    for path in sorted(csv_paths):
        basename = os.path.basename(path)          # e.g. '20250217.data.csv'
        # Take the part before the first dot: '20250217'
        trading_date_str = basename.split(".")[0]

        # Expect exactly 8-digit YYYYMMDD
        if not (trading_date_str.isdigit() and len(trading_date_str) == 8):
            # skip any other random CSVs
            continue

        try:
            trading_date = pd.to_datetime(trading_date_str, format="%Y%m%d").date()
        except Exception:
            # If parsing fails, skip this file
            continue

        # Read the day file
        df = pd.read_csv(path, na_values=["nan", "inf"])

        # Attach proper trading_date and timestamp
        df["trading_date"] = trading_date
        df["timestamp"] = pd.to_datetime(
            df["trading_date"].astype(str) + " " + df["time"]
        )

        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(
            f"CSV files were found under {EXTRACT_DIR}, "
            f"but none had valid YYYYMMDD.*.csv naming."
        )

    data = pd.concat(all_dfs, ignore_index=True)
    return data


# ----------------------------------------------------------------------
# BUILD PANEL: CM + FUT1 + FUT2 PER (timestamp, underlying)
# ----------------------------------------------------------------------

def build_panel(data_raw: pd.DataFrame) -> pd.DataFrame:
    df = data_raw.copy()

    # Cash market rows
    cm = df[df["exchange"] == "NSECM"].copy()
    cm["underlying"] = cm["name"]
    cm["expiry_date"] = pd.NaT

    # Futures rows
    fut = df[df["exchange"] == "NSEFO"].copy()

    underlyings = []
    expiries = []
    for name in fut["name"]:
        u, exp = parse_fut_name(str(name))
        underlyings.append(u)
        expiries.append(exp)

    fut["underlying"] = underlyings
    fut["expiry_date"] = expiries
    fut = fut[~fut["underlying"].isna()]  # drop unknown futures

    # Only keep contracts whose expiry is ON or AFTER the trading date
    fut_valid = fut[pd.to_datetime(fut["expiry_date"]) >= pd.to_datetime(fut["trading_date"])].copy()
    # Rank valid contracts by expiry
    contracts = (
        fut_valid[["trading_date", "underlying", "name", "expiry_date"]]
        .drop_duplicates()
        .sort_values(["trading_date", "underlying", "expiry_date"])
        .assign(rank=lambda d: d.groupby(["trading_date", "underlying"]).cumcount() + 1)
    )

    contracts["fut_role"] = np.where(
        contracts["rank"] == 1, "FUT1",
        np.where(contracts["rank"] == 2, "FUT2", None)
    )

    contracts = contracts[contracts["fut_role"].notna()]

    fut = fut.merge(
        contracts[["trading_date", "underlying", "name", "fut_role"]],
        on=["trading_date", "underlying", "name"],
        how="inner"
    )

    # Compute days_to_expiry from FUT1 expiry
    fut1_meta = (fut[fut["fut_role"] == "FUT1"]
                 [["trading_date", "underlying", "expiry_date"]]
                 .drop_duplicates())

    fut1_meta["days_to_expiry"] = (
        pd.to_datetime(fut1_meta["expiry_date"]) -
        pd.to_datetime(fut1_meta["trading_date"])
    ).dt.days

    df_all = pd.concat([cm, fut], ignore_index=True)

    df_all = df_all.merge(
        fut1_meta[["trading_date", "underlying", "expiry_date", "days_to_expiry"]],
        on=["trading_date", "underlying"],
        how="left"
    )

    # Build CM, FUT1, FUT2 panels
    cm_panel = (df_all[df_all["exchange"] == "NSECM"]
                .copy()
                .rename(columns={
                    "ltp": "cm_ltp",
                    "total_trade_qty": "cm_total_trade_qty",
                    "lot_size": "cm_lot_size"
                })
                [["timestamp", "trading_date", "underlying",
                  "cm_ltp", "cm_total_trade_qty", "cm_lot_size",
                  "days_to_expiry"]])

    fut1_panel = (df_all[df_all["fut_role"] == "FUT1"]
                  .copy()
                  .rename(columns={
                      "ltp": "fut1_ltp",
                      "total_trade_qty": "fut1_total_trade_qty",
                      "lot_size": "fut1_lot_size"
                  })
                  [["timestamp", "trading_date", "underlying",
                    "fut1_ltp", "fut1_total_trade_qty", "fut1_lot_size"]])

    fut2_panel = (df_all[df_all["fut_role"] == "FUT2"]
                  .copy()
                  .rename(columns={
                      "ltp": "fut2_ltp",
                      "total_trade_qty": "fut2_total_trade_qty",
                      "lot_size": "fut2_lot_size"
                  })
                  [["timestamp", "trading_date", "underlying",
                    "fut2_ltp", "fut2_total_trade_qty", "fut2_lot_size"]])

    panel = cm_panel.merge(
        fut1_panel, on=["timestamp", "trading_date", "underlying"], how="inner"
    )
    panel = panel.merge(
        fut2_panel, on=["timestamp", "trading_date", "underlying"], how="left"
    )

    # Sort and compute minute volumes from cumulative total_trade_qty
    panel = panel.sort_values(["underlying", "trading_date", "timestamp"])

    for col, newcol in [
        ("cm_total_trade_qty", "cm_volume"),
        ("fut1_total_trade_qty", "fut1_volume"),
        ("fut2_total_trade_qty", "fut2_volume"),
    ]:
        if col in panel.columns:
            panel[newcol] = (
                panel
                .groupby(["underlying", "trading_date"])[col]
                .diff()
                .fillna(panel[col])
                .clip(lower=0)
            )
        else:
            panel[newcol] = np.nan

    # Spreads
    panel["cm_fut1_spread"] = panel["fut1_ltp"] - panel["cm_ltp"]
    panel["fut1_fut2_spread"] = panel["fut2_ltp"] - panel["fut1_ltp"]

    # Volume ratios
    panel["cm_fut1_vol_ratio"] = panel["fut1_volume"] / panel["cm_volume"].replace(0, np.nan)
    panel["fut1_fut2_vol_ratio"] = panel["fut2_volume"] / panel["fut1_volume"].replace(0, np.nan)

    return panel


# ----------------------------------------------------------------------
# PLOTTING HELPERS
# ----------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_spreads_vs_dte(panel: pd.DataFrame):
    """
    Problem 1A:
    - cm_fut1_spread vs days_to_expiry
    - fut1_fut2_spread vs days_to_expiry
    per underlying and overall index (mean across names)
    """
    out_dir = os.path.join(OUTPUT_DIR, "A_spreads_vs_dte")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "by_underlying"))

    near = panel[panel["days_to_expiry"].between(0, NEAR_DTE_MAX)].copy()

    # Per underlying: daily mean spread vs days_to_expiry
    spread_stats = (
        near
        .groupby(["underlying", "days_to_expiry"])
        .agg(
            cm_fut1_mean=("cm_fut1_spread", "mean"),
            fut1_fut2_mean=("fut1_fut2_spread", "mean"),
            cm_fut1_std=("cm_fut1_spread", "std"),
            fut1_fut2_std=("fut1_fut2_spread", "std"),
        )
        .reset_index()
    )

    for sym in spread_stats["underlying"].unique():
        df_s = spread_stats[spread_stats["underlying"] == sym]
        plt.figure(figsize=(8, 4))
        plt.plot(df_s["days_to_expiry"], df_s["cm_fut1_mean"], marker="o", label="cm_fut1")
        plt.plot(df_s["days_to_expiry"], df_s["fut1_fut2_mean"], marker="s", label="fut1_fut2")
        plt.xlabel("Days to expiry (FUT1)")
        plt.ylabel("Mean spread")
        plt.title(f"{sym} - Mean spreads vs days_to_expiry (near expiry)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(out_dir, "by_underlying", f"{sym}_spreads_vs_dte.png")
        plt.savefig(fname)
        plt.close()

    # "Index" = cross-sectional average across underlyings at each timestamp
    idx_spreads = (near
                   .groupby(["timestamp", "days_to_expiry"])
                   .agg(
                       cm_fut1_idx=("cm_fut1_spread", "mean"),
                       fut1_fut2_idx=("fut1_fut2_spread", "mean")
                   )
                   .reset_index())

    idx_vs_dte = (idx_spreads
                  .groupby("days_to_expiry")
                  .agg(
                      cm_fut1_mean=("cm_fut1_idx", "mean"),
                      fut1_fut2_mean=("fut1_fut2_idx", "mean")
                  )
                  .reset_index())

    plt.figure(figsize=(8, 4))
    plt.plot(idx_vs_dte["days_to_expiry"], idx_vs_dte["cm_fut1_mean"], marker="o", label="cm_fut1")
    plt.plot(idx_vs_dte["days_to_expiry"], idx_vs_dte["fut1_fut2_mean"], marker="s", label="fut1_fut2")
    plt.xlabel("Days to expiry (FUT1)")
    plt.ylabel("Index mean spread")
    plt.title("Index-level mean spreads vs days_to_expiry (near expiry)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, "index_spreads_vs_dte.png")
    plt.savefig(fname)
    plt.close()


def plot_volume_ratios_vs_dte(panel: pd.DataFrame):
    """
    Problem 1B:
    - cm_fut1_vol_ratio vs days_to_expiry
    - fut1_fut2_vol_ratio vs days_to_expiry
    per underlying and overall index
    """
    out_dir = os.path.join(OUTPUT_DIR, "B_volume_ratios_vs_dte")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "by_underlying"))

    near = panel[panel["days_to_expiry"].between(0, NEAR_DTE_MAX)].copy()

    vol_stats = (near
                 .groupby(["underlying", "days_to_expiry"])
                 .agg(
                     cm_fut1_vol_mean=("cm_fut1_vol_ratio", "mean"),
                     fut1_fut2_vol_mean=("fut1_fut2_vol_ratio", "mean")
                 )
                 .reset_index())

    for sym in vol_stats["underlying"].unique():
        df_s = vol_stats[vol_stats["underlying"] == sym]
        plt.figure(figsize=(8, 4))
        plt.plot(df_s["days_to_expiry"], df_s["cm_fut1_vol_mean"], marker="o", label="FUT1/CM volume")
        plt.plot(df_s["days_to_expiry"], df_s["fut1_fut2_vol_mean"], marker="s", label="FUT2/FUT1 volume")
        plt.xlabel("Days to expiry (FUT1)")
        plt.ylabel("Mean volume ratio")
        plt.title(f"{sym} - Volume ratios vs days_to_expiry (near expiry)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(out_dir, "by_underlying", f"{sym}_vol_ratios_vs_dte.png")
        plt.savefig(fname)
        plt.close()

    # Index-level volume ratios: cross-sectional mean
    idx_vol = (near
               .groupby(["timestamp", "days_to_expiry"])
               .agg(
                   cm_fut1_vol_idx=("cm_fut1_vol_ratio", "mean"),
                   fut1_fut2_vol_idx=("fut1_fut2_vol_ratio", "mean")
               )
               .reset_index())

    idx_vs_dte = (idx_vol
                  .groupby("days_to_expiry")
                  .agg(
                      cm_fut1_vol_mean=("cm_fut1_vol_idx", "mean"),
                      fut1_fut2_vol_mean=("fut1_fut2_vol_idx", "mean")
                  )
                  .reset_index())

    plt.figure(figsize=(8, 4))
    plt.plot(idx_vs_dte["days_to_expiry"], idx_vs_dte["cm_fut1_vol_mean"], marker="o", label="FUT1/CM volume")
    plt.plot(idx_vs_dte["days_to_expiry"], idx_vs_dte["fut1_fut2_vol_mean"], marker="s", label="FUT2/FUT1 volume")
    plt.xlabel("Days to expiry (FUT1)")
    plt.ylabel("Index mean volume ratio")
    plt.title("Index-level volume ratios vs days_to_expiry (near expiry)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, "index_vol_ratios_vs_dte.png")
    plt.savefig(fname)
    plt.close()


def plot_spread_distributions(panel: pd.DataFrame):
    """
    Problem 1C:
    Distribution of spreads (cm_fut1_spread, fut1_fut2_spread)
    across names and days_to_expiry buckets.
    """
    out_dir = os.path.join(OUTPUT_DIR, "C_spread_distributions")
    ensure_dir(out_dir)

    near = panel[panel["days_to_expiry"].between(0, NEAR_DTE_MAX)].copy()

    bins = [-1, 2, 5, NEAR_DTE_MAX]
    labels = ["0-2", "3-5", "6-10"]
    near["dte_bucket"] = pd.cut(near["days_to_expiry"], bins=bins, labels=labels)

    for spread_col, label_prefix in [
        ("cm_fut1_spread", "cm_fut1"),
        ("fut1_fut2_spread", "fut1_fut2"),
    ]:
        for bucket in labels:
            df_b = near[near["dte_bucket"] == bucket]
            if df_b.empty:
                continue

            plt.figure(figsize=(8, 4))
            plt.hist(df_b[spread_col].dropna(), bins=50, density=True)
            plt.xlabel("Spread")
            plt.ylabel("Density")
            plt.title(f"{label_prefix} spread distribution - DTE bucket {bucket}")
            plt.grid(True)
            plt.tight_layout()
            fname = os.path.join(
                out_dir, f"{label_prefix}_spread_dist_dte_{bucket}.png"
            )
            plt.savefig(fname)
            plt.close()

    #boxplots of spreads vs exact days_to_expiry
    for spread_col, label_prefix in [
        ("cm_fut1_spread", "cm_fut1"),
        ("fut1_fut2_spread", "fut1_fut2"),
    ]:
        df_box = near[["days_to_expiry", spread_col]].dropna()
        if df_box.empty:
            continue

        # To keep boxplot readable, sort by DTE and convert to categories
        dtes = sorted(df_box["days_to_expiry"].unique())
        data_to_plot = [df_box[df_box["days_to_expiry"] == d][spread_col].values for d in dtes]

        plt.figure(figsize=(12, 4))
        plt.boxplot(data_to_plot, labels=dtes, showfliers=False)
        plt.xlabel("Days to expiry (FUT1)")
        plt.ylabel("Spread")
        plt.title(f"{label_prefix} spread boxplot vs days_to_expiry (near expiry)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{label_prefix}_boxplot_vs_dte.png")
        plt.savefig(fname)
        plt.close()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    ensure_dir(OUTPUT_DIR)
    extract_archive()
    print("Loading raw data...")
    data_raw = load_all_days()

    print("Building CM/FUT1/FUT2 panel...")
    panel = build_panel(data_raw)

    # panel_out_path = os.path.join(OUTPUT_DIR, "panel_problem1.parquet")
    # panel.to_parquet(panel_out_path, index=False)
    # print(f"Saved panel to {panel_out_path}")

    print("Generating Problem 1 plots...")

    print(" A) Spreads vs days_to_expiry")
    plot_spreads_vs_dte(panel)

    print(" B) Volume ratios vs days_to_expiry")
    plot_volume_ratios_vs_dte(panel)

    print(" C) Spread distributions across names and DTE buckets")
    plot_spread_distributions(panel)

if __name__ == "__main__":
    main()
