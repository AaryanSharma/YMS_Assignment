#!/user/mayank/projects/alphabench/venv/bin/python

import argparse, os, glob, re
import pandas as pd
import numpy as np

def _extract_params_from_filename(path: str):
    """
    Pull E/TP/SL from filenames like: mr_spread_rel_summary_E1.5_TP0.5_SL2.0.csv
    Returns (E, TP, SL) as floats or (None,None,None) if not found.
    """
    basename=os.path.basename(path)
    if basename.endswith(".csv"):
        basename=".".join(basename.split(".")[:-1])
    m = re.search(r"_E([0-9.]+)_TP([0-9.]+)_SL([0-9.]+)", basename)
    if not m:
        return (None, None, None)
    return tuple(float(x) for x in m.groups())

def _with_params(df: pd.DataFrame, params):
    E, TP, SL = params
    if df is None or df.empty:
        return df
    # prefer in-file columns if present; otherwise add from filename
    if 'entry_E' not in df.columns: df['entry_E'] = E
    if 'tp_off'  not in df.columns: df['tp_off']  = TP
    if 'stop_off'not in df.columns: df['stop_off']= SL
    return df

def load_all_summaries(result_folder: str, prefix="mr_spread_rel_summary"):
    paths = glob.glob(os.path.join(result_folder, f"{prefix}_E*_TP*_SL*.csv"))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            params = _extract_params_from_filename(p)
            df = _with_params(df, params)
            # Drop any duplicate TOTAL rows if you want per-underlying only aggregate later
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    if not frames:
        return pd.DataFrame()
    allsum = pd.concat(frames, ignore_index=True, sort=False)
    return allsum

def build_leaderboards(allsum: pd.DataFrame):
    """
    Produces:
      - combined_summary.csv: raw concatenation of all summaries
      - leaderboard_by_underlying.csv: best config per UNDERLYING by final_equity
      - leaderboard_overall.csv: best config globally by sum(final_equity) across underlyings
    """
    # Keep only per-underlying rows (exclude TOTAL if present)
    sum_u = allsum[allsum['UNDERLYING'] != 'TOTAL'].copy() if 'UNDERLYING' in allsum else allsum.copy()

    # Best per-underlying (max final_equity)
    per_name_idx = sum_u.groupby('UNDERLYING')['final_net_pnl'].idxmax()
    leaderboard_by_underlying = sum_u.loc[per_name_idx].sort_values('final_net_pnl', ascending=False)

    # Global score: sum final_equity across names per (E, TP, SL)
    agg = (sum_u.groupby(['entry_E','tp_off','stop_off'], as_index=False)
                 .agg(total_net_pnl=('final_net_pnl','sum'),
                      total_cost_pnl=('final_cost_pnl', 'sum'),
                      total_contracts=('total_contracts_traded','sum'),
                      total_shares=('total_shares_traded','sum')))
    leaderboard_overall = agg.sort_values('total_net_pnl', ascending=False)

    return leaderboard_by_underlying, leaderboard_overall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_folder", required=True, help="Folder containing *_summary_E*_TP*_SL*.csv")
    ap.add_argument("--prefix", default="mr_spread_rel", help="Prefix used in filenames")
    args = ap.parse_args()

    # Accept both mr_spread_rel_summary_*.csv and mr_spread_summary_*.csv
    allsum = load_all_summaries(args.result_folder, prefix=f"{args.prefix}_summary")
    if allsum.empty:
        print(f"[ERR] No summary files found under {args.result_folder} with prefix {args.prefix}_summary_*.csv")
        return

    # Save combined raw
    combined_path = os.path.join(args.result_folder, f"{args.prefix}_summary_ALL.csv")
    allsum.to_csv(combined_path, index=False)

    # Leaderboards
    lb_by_name, lb_overall = build_leaderboards(allsum)
    lb_by_name.to_csv(os.path.join(args.result_folder, f"{args.prefix}_leaderboard_by_underlying.csv"), index=False)
    lb_overall.to_csv(os.path.join(args.result_folder, f"{args.prefix}_leaderboard_overall.csv"), index=False)

    print("[OK] Wrote:")
    print(" -", combined_path)
    print(" -", os.path.join(args.result_folder, f"{args.prefix}_leaderboard_by_underlying.csv"))
    print(" -", os.path.join(args.result_folder, f"{args.prefix}_leaderboard_overall.csv"))

if __name__ == "__main__":
    main()

