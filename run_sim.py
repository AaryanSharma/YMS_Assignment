#!/user/mayank/projects/alphabench/venv/bin/python

import pandas as pd
import numpy as np
import re
import glob
import os
from math import floor

# --- new knobs ---
ROLL_WINDOW_DAYS = 60
MIN_PERIODS = 120            # require some history before signals
TRIM_Q_LOW = 1.0             # keep central 99% => drop bottom 0.5% …
TRIM_Q_HIGH = 99.0           # … and top 0.5%
WINS_Q_LOW  = 1.0     # clip bottom 0.5%
WINS_Q_HIGH = 99.0    # clip top 0.5%
NO_TRADE_WARMUP_DAYS = 30    # don't trade for first 30 calendar days


# ---------------- Parsing helpers ----------------

MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
FUT_REGEX = re.compile(r'^([A-Z0-9&.\-]+?)(\d{2})([A-Z]{3})FUT$')

from dataclasses import dataclass
from itertools import product

@dataclass(frozen=True)
class RelZParams:
    entry: float      # e.g., 1.5   (enter when |z| >= entry)
    tp_off: float     # e.g., 0.5   (exit if moves in favour by >= tp_off from entry)
    stop_off: float   # e.g., 2.0   (exit if moves against by >= stop_off from entry)

def decide_delta_relative(z: float, zmul: float, cost: float, spread_cost: float, lots: int, ready: bool, p: RelZParams) -> int:
    """
    Returns {-1,0,+1} delta lots.
    - Flat: enter when |z| >= p.entry  (short if z>=+E, long if z<=-E)
    - In position: exit 1 lot on either TP or SL, computed relative to the entry threshold.
      (No scale-ins; you still throttle to 1 lot/min elsewhere.)
    """

    _null = (0, 0)
    if not ready:
        return _null

    E, TP, SL = p.entry, p.tp_off, p.stop_off

    if lots > 0:  # long spread (entered at z <= -E)
        if z >= -E + TP:    # moved in favour by >= TP
            return -1, "TP"       # take profit
        elif z <= -E - SL:    # moved against by >= SL
            return -1, "SL"       # stop loss
        elif z <= -E and z >= -E - (SL/2):
            #if abs(TP * zmul) - cost < 1.0 * (2*cost + 2*spread_cost):
            if abs(TP * zmul) - cost < 1.0 * (6*cost):
                return _null
            return +1, "LONG"
       
        return _null

    if lots < 0:  # short spread (entered at z >= +E)
        if z <= +E - TP:    # moved in favour by >= TP
            return +1, "TP"       # take profit
        elif z >= +E + SL:    # moved against by >= SL
            return +1, "SL"      # stop loss
        elif z >= +E and z <= +E + (SL/2):
            #if abs(TP * zmul) - cost < 1.0 * (2*cost + 2*spread_cost):
            if abs(TP * zmul) - cost < 1.0 * (6*cost):
                return _null
            return -1, "SHRT"

        return _null

    # flat -> entry
    if z >= +E and z <= +E + (SL/2):
        #if abs(TP * zmul) - cost < 1.0*(2*cost + 2*spread_cost):
        if abs(TP * zmul) - cost < 1.0 * (6*cost):
            return _null
        return -1, "SHRT"  # enter short
    if z <= -E and z >= -E - (SL/2):
        #if abs(TP * zmul) - cost < 1.0*(2*cost + 2*spread_cost):
        if abs(TP * zmul) - cost < 1.0 * (6*cost):
            return _null
        return +1, "LONG"  # enter long
    return _null

def mark_fut1_expiry_eod(uni: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean column 'is_fut1_expiry_eod' to 'uni':
      True iff this row is the last timestamp of the last trading day
      observed for the current FUT1 contract (name_FUT1) for that underlying.
    Works purely from observed data (no calendar needed).
    """
    df = uni.copy()

    # Ensure we have a 'date' column
    if 'date' not in df.columns:
        df['date'] = df['timestamp'].dt.date

    # Last trading DATE for each (UNDERLYING, FUT1 contract name)
    last_date_per_contract = (
        df.groupby(['underlying', 'name_FUT1'], as_index=False)['date']
          .max()
          .rename(columns={'date': 'last_fut1_date'})
    )

    # Attach last date per current contract to each row
    df = df.merge(last_date_per_contract, on=['underlying', 'name_FUT1'], how='left')

    # Is this row on the last trading DATE for its current FUT1 contract?
    df['is_fut1_last_date'] = (df['date'] == df['last_fut1_date'])

    # EOD timestamp per (UNDERLYING, DATE)
    eod_ts_per_day = (
        df.groupby(['underlying', 'date'], as_index=False)['timestamp']
          .max()
          .rename(columns={'timestamp': 'eod_ts'})
    )
    df = df.merge(eod_ts_per_day, on=['underlying', 'date'], how='left')
    df['is_eod_row'] = (df['timestamp'] == df['eod_ts'])

    # Final flag: last day for this FUT1 contract AND end-of-day
    df['is_fut1_expiry_eod'] = df['is_fut1_last_date'] & df['is_eod_row']

    # Clean up helper columns if you like
    df = df.drop(columns=['eod_ts'])
    return df


def add_global_stats(uni: pd.DataFrame) -> pd.DataFrame:
    uni = uni.sort_values('timestamp')
    s = uni.set_index('timestamp')['spread']
    roll = s.rolling("60D", min_periods=120)

    uni['global_mean'] = roll.mean().values
    uni['global_sd']   = roll.std(ddof=0).values
    return uni.dropna(subset=['global_mean','global_sd']).reset_index(drop=True)

def parse_future_name(name: str):
    if not isinstance(name, str):
        return (None, None, None)
    m = FUT_REGEX.match(name.strip())
    if not m:
        return (None, None, None)
    underlying, yy, mon = m.groups()
    year = 2000 + int(yy)
    month = MONTHS.get(mon, None)
    return (underlying, year, month)

def load_one_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", na_values=['nan','NaN','','inf','-inf'])
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    for col in ['ltp','bid','ask','mid','spread','last_trade_qty','total_trade_amount','total_trade_qty','lot_size']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Pull YYYYMMDD from filename and build a proper timestamp
    base = os.path.basename(path)
    file_date = base.split(".")[0]  # YYYYMMDD
    ts = pd.to_datetime(file_date + " " + df['time'].astype(str), format="%Y%m%d %H:%M:%S.%f", errors='coerce')
    ts = ts.fillna(pd.to_datetime(file_date + " " + df['time'].astype(str), format="%Y%m%d %H:%M:%S", errors='coerce'))
    df['timestamp'] = ts
    df['file_date'] = file_date
    df['date'] = df['timestamp'].dt.date
    return df

def prepare_spread_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-timestamp rows with:
    ['timestamp','date','underlying','cash_ltp','name_FUT1','name_FUT2','ltp_FUT1','ltp_FUT2',
     'ttq_FUT1','ttq_FUT2','spread','month_fut1']
    """
    # --- Futures side ---
    fut = df[(df['exchange']=='NSEFO') & df['name'].str.endswith('FUT', na=False)].copy()
    parsed = fut['name'].apply(parse_future_name)
    fut[['underlying','exp_year','exp_month']] = pd.DataFrame(parsed.tolist(), index=fut.index)
    fut = fut.dropna(subset=['underlying','exp_year','exp_month','ltp','bid','ask','mid','spread','timestamp'])
    fut['exp_year'] = fut['exp_year'].astype(int)
    fut['exp_month'] = fut['exp_month'].astype(int)
    fut['expiry_key'] = fut['exp_year']*100 + fut['exp_month']

    # nearest two expiries per (timestamp, underlying)
    fut_sorted = fut.sort_values(['timestamp','underlying','expiry_key'])
    fut_top2 = fut_sorted.groupby(['timestamp','underlying'], as_index=False).head(2)
    fut_top2['fut_rank'] = fut_top2.groupby(['timestamp','underlying']).cumcount()+1

    # Pivot also carrying total_trade_qty (cumulative at that moment)
    wide = fut_top2.pivot_table(
        index=['timestamp','underlying'],
        columns='fut_rank',
        values=['name','ltp','bid','ask','mid','spread','exp_year','exp_month','total_trade_qty', "lot_size"],
        aggfunc='first'
    )
    wide.columns = [f"{a}_FUT{b}" for a,b in wide.columns]
    wide = wide.reset_index()

    # --- Cash (CM) side for normalization and lot sizing ---
    cash = df[df['exchange']=='NSECM'][['timestamp','name','ltp','mid']].rename(
            columns={'name':'underlying','ltp':'cash_ltp','mid':'cash_mid'}
    )

    merged = wide.merge(cash, on=['timestamp','underlying'], how='left')
    merged = merged.dropna(subset=['ltp_FUT1','ltp_FUT2','cash_ltp','mid_FUT1','mid_FUT2','cash_mid', 'spread_FUT1', 'spread_FUT2']).copy()

    # normalized spread
    merged['spread'] = 2 * (merged['mid_FUT2'] - merged['mid_FUT1']) / (merged['mid_FUT2'] + merged['mid_FUT1'])
    merged['month_fut1'] = merged['exp_month_FUT1'].astype(int)
    merged['date'] = merged['timestamp'].dt.date

    # Keep FUT2 cumulative traded qty column to compute EOD total later
    merged = merged.rename(columns={'total_trade_qty_FUT2': 'ttq_FUT2',
                                    'total_trade_qty_FUT1': 'ttq_FUT1'})

    return merged[['timestamp','date','underlying','cash_ltp','cash_mid',
                   'name_FUT1','name_FUT2','ltp_FUT1','ltp_FUT2', 'mid_FUT1', 'mid_FUT2', 'spread_FUT1', 'spread_FUT2',
                   'ttq_FUT1','ttq_FUT2','spread','month_fut1']]

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# (Optional) avoid thread oversubscription inside each worker
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

def _process_one_file(path: str) -> pd.DataFrame:
    """Top-level worker: read one CSV and return the prepared per-file frame."""
    df = load_one_file(path)            # your existing function
    return prepare_spread_frame(df)     # your existing function

# 60 calendar-day rolling SMA/STD per underlying (time-based window)
def _rolling_winsor_stats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('timestamp')
    s = g.set_index('timestamp')['spread']
    roll = s.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)

    def _wmean(a):
        if a.size == 0 or np.all(np.isnan(a)):
            return np.nan
        lo, hi = np.nanpercentile(a, [WINS_Q_LOW, WINS_Q_HIGH])
        b = np.clip(a, lo, hi)
        return np.nanmean(b)

    def _wstd(a):
        if a.size == 0 or np.all(np.isnan(a)):
            return np.nan
        lo, hi = np.nanpercentile(a, [WINS_Q_LOW, WINS_Q_HIGH])
        b = np.clip(a, lo, hi)
        return np.nanstd(b, ddof=0)

    def _pos_delta(a):
        # max(last - first, 0), skipping NaNs at edges
        if a.size < 2: return 0.0
        # trim NaNs from both ends
        i0 = 0
        while i0 < a.size and np.isnan(a[i0]): i0 += 1
        i1 = a.size - 1
        while i1 >= 0 and np.isnan(a[i1]): i1 -= 1
        if i1 <= i0: return 0.0
        d = a[i1] - a[i0]
        return float(d) if d > 0 else 0.0

    g['sma'] = roll.apply(_wmean, raw=True).values
    g['sd']  = roll.apply(_wstd,  raw=True).values

    s = g.set_index('timestamp')['spread_FUT1']
    roll = s.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)
    g['sma_spread_FUT1'] = roll.apply(_wmean, raw=True).values
    g['sd_spread_FUT1']  = roll.apply(_wstd,  raw=True).values

    s = g.set_index('timestamp')['spread_FUT2']
    roll = s.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)
    g['sma_spread_FUT2'] = roll.apply(_wmean, raw=True).values
    g['sd_spread_FUT2']  = roll.apply(_wstd,  raw=True).values


    # ---------- ttq deltas & spreadable volume ----------
    # Expect cumulative quantities: ttq_FUT1 / ttq_FUT2
    ttq1 = g.set_index('timestamp')['ttq_FUT1']
    ttq2 = g.set_index('timestamp')['ttq_FUT2']

    for mins in (5, 15, 30):
        win = f'{mins}T'
        # delta over window: max(last - first, 0)
        d1 = ttq1.rolling(win, min_periods=2).apply(_pos_delta, raw=True)
        d2 = ttq2.rolling(win, min_periods=2).apply(_pos_delta, raw=True)

        g[f'ttq_FUT1_{mins}'] = d1.values
        g[f'ttq_FUT2_{mins}'] = d2.values

        # spreadable volume = min of the two legs
        g[f'ttq_SPD_{mins}'] = np.minimum(g[f'ttq_FUT1_{mins}'], g[f'ttq_FUT2_{mins}'])

        # 60D winsorized stats on spreadable volume
        s_spd = g.set_index('timestamp')[f'ttq_SPD_{mins}']
        r_spd = s_spd.rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)
        g[f'ttq_SPD_{mins}_sma'] = r_spd.apply(_wmean, raw=True).values
        g[f'ttq_SPD_{mins}_sd']  = r_spd.apply(_wstd,  raw=True).values

    return g

def build_rolling_winsor_stats_per_underlying(uni: pd.DataFrame, ncores: int = 1, u_list: list = [], window: int = 1) -> pd.DataFrame:
    # Split once by underlying to avoid groupby inside workers
    groups = [g for under, g in uni.groupby('underlying') if ((not u_list) or (under in u_list)) ]

    if not groups:
        return uni

    results = []
    with ProcessPoolExecutor(max_workers=ncores) as ex:
        futures = {ex.submit(_rolling_winsor_stats, g): g for g in groups}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rolling (winsor) per underlying"):
            results.append(fut.result())

    uni_out = pd.concat(results, ignore_index=True)
    uni_out = uni_out.sort_values(['underlying','timestamp']).reset_index(drop=True)
    return uni_out


def build_universe(folder: str, ncores: int = 1, u_list: list = [], window: int = 1) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {folder}")

    frames = []

    with ProcessPoolExecutor(max_workers=ncores) as ex:
        # submit jobs
        futures = {ex.submit(_process_one_file, f): f for f in files}
        #for fut in as_completed(futures):
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building universe"):
            frames.append(fut.result())

    uni = pd.concat(frames, ignore_index=True)

    # Sort for rolling calcs
    uni = uni.sort_values(['underlying','timestamp']).reset_index(drop=True)

    uni = build_rolling_winsor_stats_per_underlying(uni, ncores, u_list)
    #uni = uni.groupby('underlying', group_keys=False).apply(_rolling_winsor_stats)

    uni['sd_spread_FUT1'] = uni['sd_spread_FUT1'].replace(0.0, np.nan)
    uni['sd_spread_FUT2'] = uni['sd_spread_FUT2'].replace(0.0, np.nan)
    uni['sd'] = uni['sd'].replace(0.0, np.nan)
    uni['z']  = (uni['spread'] - uni['sma']) / uni['sd']
    uni = uni.dropna(subset=['sma','sd','z', 'sma_spread_FUT1', 'sma_spread_FUT2', 'sd_spread_FUT1', 'sd_spread_FUT2']).reset_index(drop=True)
    
    return uni.dropna(subset=['sma','sd','z']).reset_index(drop=True)

# ---------------- Trading simulation ----------------

#def simulate_mean_reversion(uni: pd.DataFrame,
def simulate_mean_reversion(group: (str, pd.DataFrame),
                            zparams: RelZParams,
                            lot_notional_fixed: float = 800000.0,
                            max_lots: int = 40, _min: int = 15):
    """
    Returns:
      perf: per-minute MTM by underlying
      summary: final summary by underlying
      tlog: per-trade log with delta_lots
    """
    results = []
    trades = []

    shares_per_lot_dict={}

    print(group[1].columns)
    print(group[0])

    #for under, g in uni.groupby('underlying'):
    for under, g in [group]:
        g = g.sort_values('timestamp').copy()

        # --- new: do not trade for the first 30 days from first tick of this underlying
        start_trade_ts = g['timestamp'].min() + pd.Timedelta(days=NO_TRADE_WARMUP_DAYS)
        start_time = pd.Timestamp('09:20').time()
        end_time = pd.Timestamp('15:20').time()

        lots = 0
        realized = 0.0
        unrealized = 0.0

        gross_pnl  = 0
        net_pnl = 0
        cost_pnl = 0
        own_amount = 0
        spr_slpg_pnl1 = 0
        spr_slpg_pnl2 = 0
        minutes_since_last_trade = 0

        rows = []
        tlog = []
        cost =0.01 *((0.02 + (0.00173 + 0.00010) * 2 + 0.002)/2)

        for _, row in g.iterrows():
            minutes_since_last_trade += 1
            ts   = row['timestamp']
            f1   = row['mid_FUT1']
            s1   = row['spread_FUT1']
            f2   = row['mid_FUT2']
            s2   = row['spread_FUT2']
            cash = row['cash_ltp']
            z    = row['z']; sma = row['sma']; sd = row['sd']

            gmean = row['global_mean']
            gsd   = row['global_sd']

            ttq_spd = row[f'ttq_SPD_{_min}']

            ttq_spd_sma = row[f'ttq_SPD_{_min}_sma']
            ttq_spd_sd = row[f'ttq_SPD_{_min}_sd']


            try:
                shares_per_lot = shares_per_lot_dict[under]
                lot_notional = cash * shares_per_lot
            except:
                lot_notional = lot_notional_fixed
                shares_per_lot = max(1, int(floor(lot_notional / cash)))
                shares_per_lot_dict[under] = shares_per_lot


            ttq_limit = min(ttq_spd_sma, ttq_spd) * 0.25 / shares_per_lot
            if np.isnan(ttq_limit):
                ttq_limit = 0
            else:
                ttq_limit = int(ttq_limit)

            spr = (f2 - f1)
            zmul = sd * ((f1 + f2)/2) * shares_per_lot
            cost_spr = (s1 + s2) / 2
            curr_value = lots * shares_per_lot * (spr)

            # ===== FORCE CLOSE on FUT1 expiry EOD =====
            if bool(row.get('is_fut1_expiry_eod', False)) and lots != 0:
                # Close entire position immediately (ignore 1-lot/min throttle and caps)
                delta = -lots
                lots_new = 0

                # Realize P&L on full close at current price
                realized += curr_value  # closing whole position at current MTM

                own_amount += -delta * shares_per_lot * spr
                gross_pnl = own_amount + lots_new * shares_per_lot * spr
                cost_pnl += abs(delta * lot_notional) * cost * 2
                spr_slpg_pnl1 += abs(s1/2) * shares_per_lot * abs(delta) 
                spr_slpg_pnl2 += abs(s2/2) * shares_per_lot * abs(delta) 
                net_pnl = gross_pnl - cost_pnl

                minutes_since_last_trade = 0

                tlog.append({
                    'timestamp': ts, 
                    'date': row['date'],
                    'UNDERLYING': under,
                    'action': 'FORCE_CLOSE_EXPIRY_EOD',
                    'tag': 'FORCE_CLOSE_EXPIRY_EOD',
                    'delta_lots': int(delta),
                    'lots_after': int(lots_new),
                    'spr_price': spr,
                    'shares_per_lot': int(shares_per_lot),
                    'FUT1': row['name_FUT1'], 
                    'FUT2': row['name_FUT2'],
                    'entry_E': zparams.entry, 
                    'tp_off': zparams.tp_off, 
                    'stop_off': zparams.stop_off
                })
                lots = lots_new
                curr_value = 0.0
                unrealized = 0.0
                equity = realized

                # record the row and continue to next timestamp
                rows.append({
                    'timestamp': ts, 'UNDERLYING': under,
                    'lots': int(lots), 'shares_per_lot': int(shares_per_lot),
                    'FUT1': row['name_FUT1'], 'FUT2': row['name_FUT2'],
                    'ltp_FUT1': f1, 'ltp_FUT2': f2, 'cash_ltp': cash,
                    'mid_FUT1': f1, 'mid_FUT2': f2, 'cash_mid': cash,
                    'spread_norm': row['spread'], 'z': row['z'],
                    'sma': sma, 'sd': sd,
                    'gsma': gmean, 'gsd': gsd,
                    'ttq_spd': ttq_spd, 'ttq_spd_sma': ttq_spd_sma,
                    'spr_raw': spr, 'value': curr_value,
                    'realized': realized, 'unrealized': unrealized, 'equity': equity,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'cost_pnl': cost_pnl,
                    'spr_slpg_pnl1': spr_slpg_pnl1,
                    'spr_slpg_pnl2': spr_slpg_pnl2,
                    'own_amount': own_amount,
                    'ttq_FUT2': row.get('ttq_FUT2', np.nan),
                    'entry_E': zparams.entry, 
                    'tp_off': zparams.tp_off, 
                    'stop_off': zparams.stop_off
                })
                continue
            # ===== END FORCE CLOSE =====

            # --- global filter ---

            cmean1 = row['sma_spread_FUT1']
            cmean2 = row['sma_spread_FUT2']
            sd1 = row['sd_spread_FUT1']
            sd2 = row['sd_spread_FUT2']

            # --- existing signal logic gives desired delta ---
            delta = 0

            ready = (ts >= start_trade_ts) and (ts.time() >= start_time and ts.time() <= end_time)

            if s1 <= 0 or s2 <= 0:
                continue

            spread_cost = abs(s1/2 + s2/2) * shares_per_lot
            delta, tag = decide_delta_relative(z, zmul, cost * lot_notional * 2, spread_cost, lots, ready, zparams)


            # Throttle and bounds
            delta = max(-1, min(1, delta))
            #delta *= ttq_limit

            if lots + delta >  max_lots: delta = max_lots - lots
            if lots + delta < -max_lots: delta = -max_lots - lots

            spr_norm = spr / cash

            if (spr_norm < gmean - 2*gsd) and ready:
                delta = -lots
                tag = "EXIT_G_LIMIT"

            if (spr_norm < gmean - 1.75*gsd) and ready:
                if delta * lots >= 0:
                    delta = 0

            delta = max(-1, min(1, delta))

            #if spr_norm < 0.0020:
            #    delta = -lots    # skip all trades for this underlying at this ts

            if delta != 0:
                if ttq_limit == 0 or minutes_since_last_trade < _min / ttq_limit:
                    delta = 0

            if delta != 0:
                minutes_since_last_trade = 0
                trade_side = 'BUY_SPREAD' if delta > 0 else 'SELL_SPREAD'
                lots_new = lots + delta


                if np.sign(lots) != 0 and np.sign(lots) != np.sign(lots_new):
                    realized += curr_value
                elif abs(lots_new) < abs(lots):
                    realized += (abs(lots) - abs(lots_new)) * shares_per_lot * spr * np.sign(lots)

                own_amount += -delta * shares_per_lot * spr
                gross_pnl = own_amount + lots_new * shares_per_lot * spr
                #cost_pnl += abs(delta * shares_per_lot *spr) * cost * 2
                cost_pnl += abs(delta * lot_notional
                        ) * cost * 2
                spr_slpg_pnl1 +=  abs(s1/2) * shares_per_lot * abs(delta)
                spr_slpg_pnl2 +=  abs(s2/2) * shares_per_lot * abs(delta)
                net_pnl = gross_pnl - cost_pnl

                tlog.append({
                    'timestamp': ts,
                    'date': row['date'],
                    'UNDERLYING': under,
                    'action': trade_side,
                    'tag': tag,
                    'delta_lots': int(delta),
                    'lots_after': int(lots_new),
                    'spr_price': spr,
                    'shares_per_lot': int(shares_per_lot),
                    'FUT1': row['name_FUT1'],
                    'FUT2': row['name_FUT2'],
                    'entry_E': zparams.entry, 
                    'tp_off': zparams.tp_off, 
                    'stop_off': zparams.stop_off
                })
                lots = lots_new
                curr_value = lots * shares_per_lot * spr

            unrealized = curr_value
            equity = realized + unrealized


            rows.append({
                'timestamp': ts, 'date': row['date'], 'UNDERLYING': under,
                'lots': int(lots), 'shares_per_lot': int(shares_per_lot),
                'FUT1': row['name_FUT1'], 'FUT2': row['name_FUT2'],
                'ltp_FUT1': f1, 'ltp_FUT2': f2, 'cash_ltp': cash,
                'mid_FUT1': f1, 'mid_FUT2': f2, 'cash_mid': cash,
                'spread_norm': row['spread'], 'z': z, 'sma': sma, 'sd': sd,
                'gsma': gmean, 'gsd': gsd,
                'ttq_spd': ttq_spd, 'ttq_spd_sma': ttq_spd_sma,
                'spr_raw': spr, 'value': curr_value,
                'realized': realized, 'unrealized': unrealized, 'equity': equity,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'cost_pnl': cost_pnl,
                'spr_slpg_pnl1': spr_slpg_pnl1,
                'spr_slpg_pnl2': spr_slpg_pnl2,
                'own_amount': own_amount,
                'ttq_FUT2': row.get('ttq_FUT2', np.nan)  # cumulative contracts at this ts
            })

        results.append(pd.DataFrame(rows))
        trades.append(pd.DataFrame(tlog) if tlog else pd.DataFrame(columns=['timestamp','date','UNDERLYING','action','tag','delta_lots','lots_after','spr_price','shares_per_lot','FUT2','FUT1']))

    perf = pd.concat(results, ignore_index=True).sort_values(['UNDERLYING','timestamp'])
    tlog = pd.concat(trades, ignore_index=True).sort_values(['UNDERLYING','timestamp'])

    summary = (
    perf.groupby('UNDERLYING', as_index=False)
        .agg(
            start=('timestamp', 'min'),
            end=('timestamp', 'max'),
            final_realized=('realized', 'last'),
            final_unrealized=('unrealized', 'last'),
            final_equity=('equity', 'last'),
            final_gross_pnl=('gross_pnl', 'last'),
            final_net_pnl=('net_pnl', 'last'),
            final_cost_pnl=('cost_pnl', 'last'),
            final_spr_slpg1=('spr_slpg_pnl1', 'last'),
            final_spr_slpg2=('spr_slpg_pnl2', 'last'),
            final_own_amount=('own_amount', 'last'),
            # ↓ specify the column for the lambda
            max_drawdown=('net_pnl', lambda s: (s.cummax() - s).max())
        )
    )

    # --- add trade stats from tlog ---
    if not tlog.empty:
        trade_stats = (
            tlog.groupby('UNDERLYING', as_index=False)
                .agg(
                    total_contracts_traded=('delta_lots', lambda x: int(np.abs(x).sum())),
                    total_shares_traded=('delta_lots',
                        lambda x: int(np.abs((2*abs(tlog.loc[x.index, 'delta_lots']) *
                                              tlog.loc[x.index, 'shares_per_lot']).sum()))),
                    max_delta_lots=('lots_after', 'max'),
                    min_delta_lots=('lots_after', 'min')
                )
        )
    else:
        trade_stats = pd.DataFrame(columns=[
            'UNDERLYING','total_contracts_traded','total_shares_traded',
            'max_delta_lots','min_delta_lots'
        ])

    summary = summary.merge(trade_stats, on='UNDERLYING', how='left').fillna({
        'total_contracts_traded': 0,
        'total_shares_traded': 0,
        'max_delta_lots': 0,
        'min_delta_lots': 0
    })


    return perf, summary, tlog

# ---------------- New: Daily breakdown ----------------

def daily_breakdown(perf: pd.DataFrame, tlog: pd.DataFrame) -> pd.DataFrame:
    """
    Per-day, per-underlying with carry-forward of unrealized into next day's realized:
      - daily_realized_cf: realized-to-realized P&L with carry-forward
      - eod_unrealized, eod_equity
      - daily_pnl_total = eod_equity - equity(prev day)
      - lots_traded, fut2_total_contracts, lots_traded_pct_of_fut2
    """
    # Base daily aggregates
    base = (
        perf.groupby(['UNDERLYING','date'], as_index=False)
            .agg(realized_start=('realized','first'),
                 realized_end=('realized','last'),
                 eod_unrealized=('unrealized','last'),
                 eod_equity=('equity','last'))
            .sort_values(['UNDERLYING','date'])
    )

    # Lag (yesterday) values per underlying
    base['eod_unrealized_prev'] = base.groupby('UNDERLYING')['eod_unrealized'].shift(1).fillna(0.0)
    base['eod_equity_prev']     = base.groupby('UNDERLYING')['eod_equity'].shift(1).fillna(0.0)

    # Carry-forward: treat yesterday's EOD unrealized as realized at today's open
    base['realized_start_cf'] = base['realized_start'] + base['eod_unrealized_prev']
    base['daily_realized_cf'] = base['realized_end'] - base['realized_start_cf']

    # Total daily MTM P&L (equity change)
    base['daily_pnl_total'] = base['eod_equity'] - base['eod_equity_prev']

    # Lots traded today
    lots_day = (
        tlog.groupby(['UNDERLYING','date'], as_index=False)
            .agg(lots_traded=('delta_lots', lambda x: int(np.abs(x).sum())))
    )

    # FUT2 EOD contracts (use daily max of cumulative ttq_FUT2)
    fut2_eod = (
        perf.groupby(['UNDERLYING','date'], as_index=False)
            .agg(fut2_total_contracts=('ttq_FUT2','max'))
    )

    daily = (base
             .merge(lots_day, on=['UNDERLYING','date'], how='left')
             .merge(fut2_eod, on=['UNDERLYING','date'], how='left'))

    daily['lots_traded'] = daily['lots_traded'].fillna(0).astype(int)
    daily['lots_traded_pct_of_fut2'] = np.where(
        (daily['fut2_total_contracts'] > 0) & (~daily['fut2_total_contracts'].isna()),
        100.0 * daily['lots_traded'] / daily['fut2_total_contracts'],
        np.nan
    )

    return daily[['UNDERLYING','date',
                  'daily_realized_cf','eod_unrealized','eod_equity','daily_pnl_total',
                  'lots_traded','fut2_total_contracts','lots_traded_pct_of_fut2']] \
           .sort_values(['UNDERLYING','date'])

# ---------------- Driver ----------------


def run_relz_grid(uni: pd.DataFrame,
                  entry_list=(1.0, 1.25, 1.5, 1.75, 2.0),
                  tp_off_list=(0.5, 0.75, 1.0),
                  stop_off_list=(1.5, 2.0, 2.5),
                  lot_notional=800000.0, max_lots=40,
                  save_dir=None, prefix="mr_spread"):
    all_perf, all_summary, all_trades = [], [], []

    for E, TP, SL in product(entry_list, tp_off_list, stop_off_list):
        zp = RelZParams(entry=E, tp_off=TP, stop_off=SL)
        perf, summary, tlog = simulate_mean_reversion(
            uni, zparams=zp, lot_notional_fixed=lot_notional, max_lots=max_lots
        )
        cfg = f"E{E}_TP{TP}_SL{SL}"
        for df in (perf, summary, tlog):
            df['cfg'] = cfg

        all_perf.append(perf)
        all_summary.append(summary)
        all_trades.append(tlog)

        if save_dir:
            #perf.to_csv(f"{save_dir}/{prefix}_perminute_{cfg}.csv", index=False)
            tlog.to_csv(f"{save_dir}/{prefix}_trades_{cfg}.csv", index=False)
            summary.to_csv(f"{save_dir}/{prefix}_summary_{cfg}.csv", index=False)

    perf_cat = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()
    tlog_cat = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summ_cat = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    if save_dir:
        #perf_cat.to_csv(f"{save_dir}/{prefix}_perminute_ALL.csv", index=False)
        tlog_cat.to_csv(f"{save_dir}/{prefix}_trades_ALL.csv", index=False)
        summ_cat.to_csv(f"{save_dir}/{prefix}_summary_ALL.csv", index=False)

    return perf_cat, summ_cat, tlog_cat

def run_sim(folder: str,
            out_prefix: str = "mr_spread",
            save_csv: bool = True):
    uni = build_universe(folder)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)
    perf, summary, trades = simulate_mean_reversion(uni, RelZParams(entry=1.5, tp_off=0.5, stop_off=1.5))

    # New daily breakdown
    daily = daily_breakdown(perf, trades)

    if save_csv:
        perf.to_csv(os.path.join(folder, f"{out_prefix}_perminute.csv"), index=False)
        summary.to_csv(os.path.join(folder, f"{out_prefix}_summary.csv"), index=False)
        trades.to_csv(os.path.join(folder, f"{out_prefix}_trades.csv"), index=False)
        daily.to_csv(os.path.join(folder, f"{out_prefix}_daily.csv"), index=False)

    return perf, summary, trades, daily

def run_sim_grid(folder: str,
            out_prefix: str = "mr_spread_rel",
            save_csv: bool = True):
    uni = build_universe(folder)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)

    perf_all, summary_all, trades_all = run_relz_grid(
        uni,
        entry_list=[1.0,1.25,1.5,1.75,2.0],
        tp_off_list=[0.25,0.5,0.75,1.0,1.25,1.5],
        stop_off_list=[0.75,1.0,1.25,1.5],
        save_dir="./results/",
        prefix=out_prefix
    )

    return perf_all, summary_all, trades_all

# Example:
#perf, summary, trades = run_sim_grid("./customdata")

# ==== CLI driver (single-config run) =========================================
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Mean-reversion spread sim (single zparam run).")
    parser.add_argument("--entry", type=float, required=True, help="Entry z threshold E (e.g., 1.5)")
    parser.add_argument("--tp_off", type=float, required=True, help="TP offset from entry (e.g., 0.5)")
    parser.add_argument("--stop_off", type=float, required=True, help="STOP offset from entry (e.g., 2.0)")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to input CSV folder")
    parser.add_argument("--result_folder", type=str, required=True, help="Folder to write result CSVs")
    # Optional knobs (keep defaults from your script):
    parser.add_argument("--lot_notional", type=float, default=800000.0)
    parser.add_argument("--max_lots", type=int, default=40)
    parser.add_argument("--prefix", type=str, default="mr_spread_rel")
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--underlying", type=str, default="")
    parser.add_argument("--window", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.result_folder, exist_ok=True)

    # === Build universe (reuse your existing pipeline) ===
    # uni = build_universe(args.data_folder)
    # uni = add_underlying_ema_stats(uni)   # EMA halflife=5D per earlier patch
    # uni = add_global_ema_stats(uni)       # global EMA stats
    # uni = mark_fut1_expiry_eod(uni)       # expiry day EOD flag
    # NOTE: Keep your existing implementations; the above line-up is illustrative.

    # If your code already has a helper that builds uni, call it here:

    import datetime
    tp = datetime.datetime.now()
    uni = build_universe(args.data_folder, args.ncores, [args.underlying] if args.underlying else [])
    
    tdiff  = datetime.datetime.now() - tp
    print(tdiff, "build_universe done")
    #uni = add_underlying_ema_stats(uni)
    #uni = add_global_ema_stats(uni)
    uni = add_global_stats(uni)
    uni = mark_fut1_expiry_eod(uni)

    tdiff  = datetime.datetime.now() - tp
    print(tdiff, "uni done")

    # === Sim run for this single zparam config ===
    zp = RelZParams(entry=args.entry, tp_off=args.tp_off, stop_off=args.stop_off)

    groups = [(under, g) for under, g in uni.groupby('underlying')]

    results = []
    with ProcessPoolExecutor(max_workers=args.ncores) as ex:
        futures = {ex.submit(simulate_mean_reversion, group=g, zparams=zp, lot_notional_fixed=args.lot_notional, max_lots=args.max_lots): g for g in groups}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Simulation Per Underlying {zp}"):
            results.append(fut.result())

    # --- concatenate outputs ---
    if results:
        perf   = pd.concat([r[0] for r in results if r is not None and len(r) >= 1], ignore_index=True)
        summary= pd.concat([r[1] for r in results if r is not None and len(r) >= 2], ignore_index=True)
        trades = pd.concat([r[2] for r in results if r is not None and len(r) >= 3], ignore_index=True)
        summary = summary.sort_values('final_net_pnl', ascending=False)

        # --- add TOTAL row ---
        total_row = {
            'UNDERLYING': 'TOTAL',
            'start': summary['start'].min(),
            'end': summary['end'].max(),
            'final_realized': summary['final_realized'].sum(),
            'final_unrealized': summary['final_unrealized'].sum(),
            'final_equity': summary['final_equity'].sum(),
            'final_gross_pnl': summary['final_gross_pnl'].sum(),
            'final_net_pnl': summary['final_net_pnl'].sum(),
            'final_cost_pnl': summary['final_cost_pnl'].sum(),
            'final_spr_slpg1': summary['final_spr_slpg1'].sum(),
            'final_spr_slpg2': summary['final_spr_slpg2'].sum(),
            'final_own_amount': summary['final_own_amount'].sum(),
            'max_drawdown': summary['max_drawdown'].max(),
            'total_contracts_traded': summary['total_contracts_traded'].sum(),
            'total_shares_traded': summary['total_shares_traded'].sum(),
            'max_delta_lots': summary['max_delta_lots'].max(),
            'min_delta_lots': summary['min_delta_lots'].min()
        }
        summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    else:
        perf = pd.DataFrame()
        summary = pd.DataFrame()
        trades = pd.DataFrame()

    #perf, summary, trades = simulate_mean_reversion(
    #    uni, zparams=zp, lot_notional_fixed=args.lot_notional, max_lots=args.max_lots
    #)

    tdiff  = datetime.datetime.now() - tp
    print(tdiff, "simulation done")
    # stamp parameters (redundant if you already stamp inside simulate)
    for df in (perf, summary, trades):
        df["entry_E"] = args.entry
        df["tp_off"] = args.tp_off
        df["stop_off"] = args.stop_off

    # Add TOTAL row (you already built this earlier; if not, do it here)
    # -- if you already add TOTAL row in summary upstream, skip this block --
    if "UNDERLYING" in summary.columns and "TOTAL" not in summary["UNDERLYING"].values:
        total_row = {
            'UNDERLYING': 'TOTAL',
            'start': summary['start'].min() if 'start' in summary else pd.NaT,
            'end': summary['end'].max() if 'end' in summary else pd.NaT,
            'final_realized': summary.get('final_realized', pd.Series([0])).sum(),
            'final_unrealized': summary.get('final_unrealized', pd.Series([0])).sum(),
            'final_equity': summary.get('final_equity', pd.Series([0])).sum(),
            'max_drawdown': summary.get('max_drawdown', pd.Series([0])).max(),
            'total_contracts_traded': summary.get('total_contracts_traded', pd.Series([0])).sum(),
            'total_shares_traded': summary.get('total_shares_traded', pd.Series([0])).sum(),
            'max_delta_lots': summary.get('max_delta_lots', pd.Series([0])).max(),
            'min_delta_lots': summary.get('min_delta_lots', pd.Series([0])).min(),
            'entry_E': args.entry, 'tp_off': args.tp_off, 'stop_off': args.stop_off
        }
        summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    # === Save with parameterized filenames ===
    tag = f"E{args.entry}_TP{args.tp_off}_SL{args.stop_off}"
    perf.to_csv(os.path.join(args.result_folder, f"{args.prefix}_perminute_{tag}.csv"), index=False)
    trades.to_csv(os.path.join(args.result_folder, f"{args.prefix}_trades_{tag}.csv"), index=False)
    summary.to_csv(os.path.join(args.result_folder, f"{args.prefix}_summary_{tag}.csv"), index=False)

    print(f"[OK] Saved results for {tag} in {args.result_folder}")

