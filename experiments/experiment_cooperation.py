"""Parameter-sweep utilities for cooperation-percentage experiments."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from neg_incomplete_info.transition_matrix import *
import time
import pandas as pd
import pickle

def _fmt_secs(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = int(sec % 60)
    if h > 0:
        return f'{h:d}h {m:02d}m {s:02d}s'
    if m > 0:
        return f'{m:d}m {s:02d}s'
    return f'{s:d}s'

class ProgressPrinter:
    """Parameter-sweep utilities for cooperation-percentage experiments."""

    def __init__(self, total: int, *, time_interval: float=5.0, step_interval: int=20):
        self.total = int(total)
        self.time_interval = float(time_interval)
        self.step_interval = int(step_interval)
        self.start_t = time.time()
        self.last_print_t = self.start_t
        self.last_print_step = 0
        self.done = 0

    def update(self, inc: int=1, *, prefix: str='', msg: str=''):
        self.done += int(inc)
        now = time.time()
        need_print = False
        if now - self.last_print_t >= self.time_interval:
            need_print = True
        if self.done - self.last_print_step >= self.step_interval:
            need_print = True
        if self.done >= self.total:
            need_print = True
        if not need_print:
            return
        elapsed = now - self.start_t
        rate = self.done / elapsed if elapsed > 1e-09 else 0.0
        remain = (self.total - self.done) / rate if rate > 1e-09 else float('inf')
        pct = 100.0 * self.done / self.total if self.total > 0 else 100.0
        eta = _fmt_secs(remain) if remain != float('inf') else '?'
        print(f'{prefix}[{self.done}/{self.total} | {pct:6.2f}%] elapsed={_fmt_secs(elapsed)} eta={eta}  {msg}')
        self.last_print_t = now
        self.last_print_step = self.done

def mean_cooperation_percentage_cached(*, cache: dict, n_players: int, d2: int, d3: int, r_over_n: float, rate: float, repeats: int):
    """Parameter-sweep utilities for cooperation-percentage experiments."""
    key = (int(d2), int(d3), float(r_over_n), float(rate))
    if key in cache:
        return cache[key]
    games = generate_games(n_players=n_players, graph_type='mixed', structure_spec={'d2': d2, 'd3': d3, 'allow_approx': True}, seed=0)
    payoffs = generate_payoffs(games, r_over_n=r_over_n)
    num_strategies = [2] * n_players
    vals = []
    for _ in range(repeats):
        san = san_generate(games=games, rate=rate, seed=None)
        transition = get_transition_matrix_policy(games, payoffs, num_strategies, san, update_rule=2)
        coop = expected_cooperation_percentage_calculation(transition, num_strategies)
        vals.append(coop)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if repeats >= 2 else 0.0
    cache[key] = (mean, std)
    return (mean, std)

def _pick_six_keys(keys):
    """Parameter-sweep utilities for cooperation-percentage experiments."""
    keys = list(keys)
    if len(keys) <= 6:
        return keys
    idx = np.linspace(0, len(keys) - 1, 6)
    idx = np.round(idx).astype(int)
    picked = []
    seen = set()
    for i in idx:
        k = keys[i]
        if k not in seen:
            picked.append(k)
            seen.add(k)
    j = 0
    while len(picked) < 6 and j < len(keys):
        if keys[j] not in seen:
            picked.append(keys[j])
            seen.add(keys[j])
        j += 1
    return picked[:6]

def _cache_to_dataframe(cache):
    """Parameter-sweep utilities for cooperation-percentage experiments."""
    rows = []
    for (d2, d3, ron, rate), val in cache.items():
        mean = None
        std = None
        raw_len = None
        try:
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                mean = float(val[0])
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    std = float(val[1])
                except Exception:
                    if hasattr(val[1], '__len__'):
                        raw_len = len(val[1])
        except Exception:
            pass
        rows.append({'d2': int(d2), 'd3': int(d3), 'r_over_n': float(ron), 'rate': float(rate), 'mean': mean, 'std': std, 'raw_len': raw_len})
    return pd.DataFrame(rows)

def run_and_plot_cached(*, n_players=10, repeats=10, d2_fixed=6, d3_fixed=6, r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), r_over_n_fixed=0.75, rate_fixed=0.3, d2_grid=np.arange(1, 9), d3_grid=np.arange(1, 11), progress_time_interval=5.0, progress_step_interval=10, save_pickle_path='./cache_full.pkl', save_csv_path='./cache_full.csv', lines_per_subplot=6):
    param_points = set()
    for rate in rate_grid:
        for ron in r_over_n_grid:
            param_points.add((d2_fixed, d3_fixed, float(ron), float(rate)))
    for d3 in d3_grid:
        for d2 in d2_grid:
            param_points.add((int(d2), int(d3), float(r_over_n_fixed), float(rate_fixed)))
    for d2 in d2_grid:
        for d3 in d3_grid:
            param_points.add((int(d2), int(d3), float(r_over_n_fixed), float(rate_fixed)))
    param_points = sorted(param_points)
    total = len(param_points)
    print(f'[INFO] Unique parameter points = {total}')
    print(f'[INFO] SAN repeats per point = {repeats}')
    pp = ProgressPrinter(total, time_interval=progress_time_interval, step_interval=progress_step_interval)
    cache = {}
    for d2, d3, ron, rate in param_points:
        mean_cooperation_percentage_cached(cache=cache, n_players=n_players, d2=d2, d3=d3, r_over_n=ron, rate=rate, repeats=repeats)
        pp.update(prefix='[CALC] ', msg=f'd2={d2}, d3={d3}, r/n={ron:.3f}, rate={rate:.3f}')
    print('[INFO] All parameter points computed. Saving data...')
    if save_pickle_path:
        with open(save_pickle_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f'[SAVE] cache pickle -> {save_pickle_path}')
    if save_csv_path:
        df = _cache_to_dataframe(cache)
        df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
        print(f'[SAVE] cache csv -> {save_csv_path}')
    A = {float(rate): [] for rate in rate_grid}
    for rate in rate_grid:
        for ron in r_over_n_grid:
            A[float(rate)].append(cache[d2_fixed, d3_fixed, float(ron), float(rate)][0])
    B = {float(ron): [] for ron in r_over_n_grid}
    for ron in r_over_n_grid:
        for rate in rate_grid:
            B[float(ron)].append(cache[d2_fixed, d3_fixed, float(ron), float(rate)][0])
    C = {int(d3): [] for d3 in d3_grid}
    for d3 in d3_grid:
        for d2 in d2_grid:
            C[int(d3)].append(cache[int(d2), int(d3), float(r_over_n_fixed), float(rate_fixed)][0])
    D = {int(d2): [] for d2 in d2_grid}
    for d2 in d2_grid:
        for d3 in d3_grid:
            D[int(d2)].append(cache[int(d2), int(d3), float(r_over_n_fixed), float(rate_fixed)][0])
    data_dict = {'A': A, 'B': B, 'C': C, 'D': D}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    ax1, ax2, ax3, ax4 = axs.ravel()
    A_keys = sorted(A.keys())
    B_keys = sorted(B.keys())
    C_keys = sorted(C.keys())
    D_keys = sorted(D.keys())
    pickA = _pick_six_keys(A_keys) if lines_per_subplot == 6 else _pick_six_keys(A_keys)[:lines_per_subplot]
    pickB = _pick_six_keys(B_keys) if lines_per_subplot == 6 else _pick_six_keys(B_keys)[:lines_per_subplot]
    pickC = _pick_six_keys(C_keys) if lines_per_subplot == 6 else _pick_six_keys(C_keys)[:lines_per_subplot]
    pickD = _pick_six_keys(D_keys) if lines_per_subplot == 6 else _pick_six_keys(D_keys)[:lines_per_subplot]
    for rate in pickA:
        ax1.plot(r_over_n_grid, A[rate], marker='o', label=f'rate={rate:.2f}')
    ax1.set_title('(a) coop vs r/n')
    ax1.set_xlabel('r_over_n')
    ax1.set_ylabel('Cooperation percentage')
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8, frameon=False)
    for ron in pickB:
        ax2.plot(rate_grid, B[ron], marker='o', label=f'r/n={ron:.2f}')
    ax2.set_title('(b) coop vs rate')
    ax2.set_xlabel('rate')
    ax2.set_ylabel('Cooperation percentage')
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=8, frameon=False)
    for d3 in pickC:
        ax3.plot(d2_grid, C[d3], marker='o', label=f'd3={d3}')
    ax3.set_title('(c) coop vs d2')
    ax3.set_xlabel('d2')
    ax3.set_ylabel('Cooperation percentage')
    ax3.set_ylim(-0.02, 1.02)
    ax3.legend(fontsize=8, frameon=False)
    for d2 in pickD:
        ax4.plot(d3_grid, D[d2], marker='o', label=f'd2={d2}')
    ax4.set_title('(d) coop vs d3')
    ax4.set_xlabel('d3')
    ax4.set_ylabel('Cooperation percentage')
    ax4.set_ylim(-0.02, 1.02)
    ax4.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()
    return (fig, cache, data_dict)

def _pick_six_keys(keys):
    keys = list(keys)
    if len(keys) <= 6:
        return keys
    idx = np.linspace(0, len(keys) - 1, 6)
    idx = np.round(idx).astype(int)
    picked, seen = ([], set())
    for i in idx:
        k = keys[i]
        if k not in seen:
            picked.append(k)
            seen.add(k)
    j = 0
    while len(picked) < 6 and j < len(keys):
        if keys[j] not in seen:
            picked.append(keys[j])
            seen.add(keys[j])
        j += 1
    return picked[:6]

def _nearest_value(values, target):
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return None
    return float(vals[np.argmin(np.abs(vals - float(target)))])

def _load_csv_unique(csv_path: str) -> pd.DataFrame:
    """Parameter-sweep utilities for cooperation-percentage experiments."""
    df = pd.read_csv(csv_path)
    df['d2'] = df['d2'].astype(int)
    df['d3'] = df['d3'].astype(int)
    df['r_over_n'] = df['r_over_n'].astype(float)
    df['rate'] = df['rate'].astype(float)
    df['mean'] = df['mean'].astype(float)
    if 'std' in df.columns:
        df['std'] = pd.to_numeric(df['std'], errors='coerce')
    agg = {'mean': 'mean'}
    if 'std' in df.columns:
        agg['std'] = 'mean'
    dfu = df.groupby(['d2', 'd3', 'r_over_n', 'rate'], as_index=False).agg(agg).sort_values(['d2', 'd3', 'r_over_n', 'rate']).reset_index(drop=True)
    return dfu

def plot_2x2_from_csv_like_cache(csv_path: str, *, d2_fixed=6, d3_fixed=6, r_over_n_fixed=0.75, rate_fixed=0.3, d2_grid=np.arange(1, 9), d3_grid=np.arange(1, 11), r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), lines_per_subplot=6, y_margin_ratio=0.06, out_path=None, show=True):
    """Parameter-sweep utilities for cooperation-percentage experiments."""
    df = _load_csv_unique(csv_path)
    ron_actual = _nearest_value(df['r_over_n'].unique(), r_over_n_fixed)
    rate_actual = _nearest_value(df['rate'].unique(), rate_fixed)
    df_ab = df[(df['d2'] == int(d2_fixed)) & (df['d3'] == int(d3_fixed))].copy()
    if df_ab.empty:
        raise ValueError(f'No rows for fixed (d2={d2_fixed}, d3={d3_fixed}).')
    A = {float(rate): [] for rate in rate_grid}
    for rate in rate_grid:
        for ron in r_over_n_grid:
            sub = df_ab[np.isclose(df_ab['rate'], float(rate)) & np.isclose(df_ab['r_over_n'], float(ron))]
            A[float(rate)].append(float(sub['mean'].iloc[0]) if not sub.empty else np.nan)
    B = {float(ron): [] for ron in r_over_n_grid}
    for ron in r_over_n_grid:
        for rate in rate_grid:
            sub = df_ab[np.isclose(df_ab['r_over_n'], float(ron)) & np.isclose(df_ab['rate'], float(rate))]
            B[float(ron)].append(float(sub['mean'].iloc[0]) if not sub.empty else np.nan)
    df_cd = df[np.isclose(df['r_over_n'], ron_actual) & np.isclose(df['rate'], rate_actual)].copy()
    if df_cd.empty:
        raise ValueError(f'No rows for (r_over_n≈{r_over_n_fixed}, rate≈{rate_fixed}). Nearest in CSV is (r_over_n={ron_actual}, rate={rate_actual}).')
    C = {int(d3): [] for d3 in d3_grid}
    for d3 in d3_grid:
        for d2 in d2_grid:
            sub = df_cd[(df_cd['d2'] == int(d2)) & (df_cd['d3'] == int(d3))]
            C[int(d3)].append(float(sub['mean'].iloc[0]) if not sub.empty else np.nan)
    D = {int(d2): [] for d2 in d2_grid}
    for d2 in d2_grid:
        for d3 in d3_grid:
            sub = df_cd[(df_cd['d2'] == int(d2)) & (df_cd['d3'] == int(d3))]
            D[int(d2)].append(float(sub['mean'].iloc[0]) if not sub.empty else np.nan)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    ax1, ax2, ax3, ax4 = axs.ravel()
    pickA = _pick_six_keys(sorted(A.keys())) if lines_per_subplot == 6 else _pick_six_keys(sorted(A.keys()))[:lines_per_subplot]
    pickB = _pick_six_keys(sorted(B.keys())) if lines_per_subplot == 6 else _pick_six_keys(sorted(B.keys()))[:lines_per_subplot]
    pickC = _pick_six_keys(sorted(C.keys())) if lines_per_subplot == 6 else _pick_six_keys(sorted(C.keys()))[:lines_per_subplot]
    pickD = _pick_six_keys(sorted(D.keys())) if lines_per_subplot == 6 else _pick_six_keys(sorted(D.keys()))[:lines_per_subplot]
    for rate in pickA:
        ax1.plot(r_over_n_grid, A[rate], marker='o', label=f'$R$={rate:.2f}')
    ax1.set_title('(a) $d_2=6,\\ d_3=6$')
    ax1.set_xlabel('$m$')
    ax1.set_ylabel('Cooperation percentage')
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8, frameon=False)
    for ron in pickB:
        ax2.plot(rate_grid, B[ron], marker='o', label=f'$m={ron:.2f}$')
    ax2.set_title('(b) $d_2 = 6,\\ d_3 = 6$')
    ax2.set_xlabel('$R$')
    ax2.set_ylabel('Cooperation percentage')
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=8, frameon=False)
    for d3 in pickC:
        ax3.plot(d2_grid, C[d3], marker='o', label=f'$d_3={d3}$')
    ax3.set_title(f'(c) $m={ron_actual:.3g},\\ R={rate_actual:.3g}$')
    ax3.set_xlabel('$d_2$')
    ax3.set_ylabel('Cooperation percentage')
    ax3.legend(fontsize=8, frameon=False)
    vals_c = np.array([v for d3 in pickC for v in C[d3] if not np.isnan(v)], dtype=float)
    if vals_c.size > 0:
        ymin, ymax = (float(vals_c.min()), float(vals_c.max()))
        span = max(1e-12, ymax - ymin)
        ax3.set_ylim(ymin - y_margin_ratio * span, ymax + y_margin_ratio * span)
    for d2 in pickD:
        ax4.plot(d3_grid, D[d2], marker='o', label=f'$d_2={d2}$')
    ax4.set_title(f'(d) $m={ron_actual:.3g},\\ R={rate_actual:.3g}$')
    ax4.set_xlabel('$d_3$')
    ax4.set_ylabel('Cooperation percentage')
    ax4.legend(fontsize=8, frameon=False)
    vals_d = np.array([v for d2 in pickD for v in D[d2] if not np.isnan(v)], dtype=float)
    if vals_d.size > 0:
        ymin, ymax = (float(vals_d.min()), float(vals_d.max()))
        span = max(1e-12, ymax - ymin)
        ax4.set_ylim(ymin - y_margin_ratio * span, ymax + y_margin_ratio * span)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (fig, {'A': A, 'B': B, 'C': C, 'D': D})

def heatmaps_from_csv_like_cache(csv_path: str, *, d2_fixed=6, d3_fixed=6, r_over_n_fixed=0.75, rate_fixed=0.3, out_path=None, show=True, figsize=(10.8, 4.2)):
    df = _load_csv_unique(csv_path)
    ron_actual = _nearest_value(df['r_over_n'].unique(), r_over_n_fixed)
    rate_actual = _nearest_value(df['rate'].unique(), rate_fixed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sub1 = df[(df['d2'] == int(d2_fixed)) & (df['d3'] == int(d3_fixed))].copy()
    if sub1.empty:
        raise ValueError(f'No rows for fixed (d2={d2_fixed}, d3={d3_fixed}).')
    rons = np.sort(sub1['r_over_n'].unique())
    rates = np.sort(sub1['rate'].unique())
    Z1 = sub1.pivot_table(index='rate', columns='r_over_n', values='mean', aggfunc='mean').reindex(index=rates, columns=rons).to_numpy()
    Z1m = np.ma.masked_invalid(Z1)
    im1 = ax1.imshow(Z1m, origin='lower', aspect='auto', extent=[rons.min(), rons.max(), rates.min(), rates.max()], interpolation='nearest')
    ax1.set_title(f'(a) $d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax1.set_xlabel('$m$')
    ax1.set_ylabel('$R$')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Cooperation percentage')
    sub2 = df[np.isclose(df['r_over_n'], ron_actual) & np.isclose(df['rate'], rate_actual)].copy()
    if sub2.empty:
        raise ValueError(f'No rows near (r_over_n={r_over_n_fixed}, rate={rate_fixed}). Nearest in CSV is (r_over_n={ron_actual}, rate={rate_actual}).')
    d2s = np.sort(sub2['d2'].unique())
    d3s = np.sort(sub2['d3'].unique())
    Z2 = sub2.pivot_table(index='d3', columns='d2', values='mean', aggfunc='mean').reindex(index=d3s, columns=d2s).to_numpy()
    Z2m = np.ma.masked_invalid(Z2)
    im2 = ax2.imshow(Z2m, origin='lower', aspect='auto', extent=[d2s.min(), d2s.max(), d3s.min(), d3s.max()], interpolation='nearest')
    ax2.set_title(f'(b) $m$={ron_actual:.3g}, $R$={rate_actual:.3g}')
    ax2.set_xlabel('$d_2$')
    ax2.set_ylabel('$d_3$')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Cooperation percentage')
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
