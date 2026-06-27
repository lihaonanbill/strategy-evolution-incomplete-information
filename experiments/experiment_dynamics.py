"""Parameter-sweep utilities for deviate-score and convergence-step experiments."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neg_incomplete_info.transition_matrix import generate_games, generate_payoffs, san_generate, get_transition_matrix_policy, deviate_score, mean_convergent_steps

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

def _set_tight_ylim(ax, ys, *, margin_ratio: float=0.06):
    ys = np.asarray([y for y in ys if np.isfinite(y)], dtype=float)
    if ys.size == 0:
        return
    ymin, ymax = (float(ys.min()), float(ys.max()))
    span = max(1e-12, ymax - ymin)
    ax.set_ylim(ymin - margin_ratio * span, ymax + margin_ratio * span)

def _nearest_value(values, target):
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return None
    return float(vals[np.argmin(np.abs(vals - float(target)))])

def _load_csv_unique(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {'d2', 'd3', 'r_over_n', 'rate'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'CSV missing columns: {sorted(missing)}. Got: {list(df.columns)}')
    df = df.copy()
    df['d2'] = df['d2'].astype(int)
    df['d3'] = df['d3'].astype(int)
    df['r_over_n'] = df['r_over_n'].astype(float)
    df['rate'] = df['rate'].astype(float)
    for col in ['dev_mean', 'dev_std', 'steps_mean', 'steps_std']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    df = df.groupby(['d2', 'd3', 'r_over_n', 'rate'], as_index=False).agg({'dev_mean': 'mean', 'dev_std': 'mean', 'steps_mean': 'mean', 'steps_std': 'mean'}).sort_values(['d2', 'd3', 'r_over_n', 'rate']).reset_index(drop=True)
    return df

def mean_metrics_cached(*, cache: dict, n_players: int, d2: int, d3: int, r_over_n: float, rate: float, repeats: int, baseline_rate: float=1.0, update_rule: int=2, base_seed: int=0):
    key = (int(d2), int(d3), float(r_over_n), float(rate))
    if key in cache:
        return cache[key]
    games = generate_games(n_players=int(n_players), graph_type='mixed', structure_spec={'d2': int(d2), 'd3': int(d3), 'allow_approx': True}, seed=0)
    payoffs = generate_payoffs(games, r_over_n=float(r_over_n))
    num_strategies = [2] * int(n_players)
    dev_vals = []
    step_vals = []
    for i in range(int(repeats)):
        san_base = san_generate(games=games, rate=float(baseline_rate), seed=None)
        san_var = san_generate(games=games, rate=float(rate), seed=None)
        baseline_transition = get_transition_matrix_policy(games, payoffs, num_strategies, san_base, update_rule=int(update_rule))
        variant_transition = get_transition_matrix_policy(games, payoffs, num_strategies, san_var, update_rule=int(update_rule))
        dev_vals.append(float(deviate_score(baseline_transition, variant_transition)))
        step_vals.append(float(mean_convergent_steps(variant_transition)))
    dev_mean = float(np.mean(dev_vals))
    dev_std = float(np.std(dev_vals, ddof=1)) if repeats >= 2 else 0.0
    steps_mean = float(np.mean(step_vals))
    steps_std = float(np.std(step_vals, ddof=1)) if repeats >= 2 else 0.0
    cache[key] = (dev_mean, dev_std, steps_mean, steps_std)
    return cache[key]

def _cache_to_dataframe(cache: dict) -> pd.DataFrame:
    rows = []
    for (d2, d3, ron, rate), val in cache.items():
        dev_mean, dev_std, steps_mean, steps_std = val
        rows.append({'d2': int(d2), 'd3': int(d3), 'r_over_n': float(ron), 'rate': float(rate), 'dev_mean': float(dev_mean), 'dev_std': float(dev_std), 'steps_mean': float(steps_mean), 'steps_std': float(steps_std)})
    return pd.DataFrame(rows)

def run_and_plot_cached(*, n_players: int=10, repeats: int=10, d2_fixed: int=6, d3_fixed: int=6, r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), r_over_n_fixed: float=0.75, rate_fixed: float=0.3, d2_grid=np.arange(1, 9), d3_grid=np.arange(1, 11), baseline_rate: float=1.0, update_rule: int=2, base_seed: int=0, progress_time_interval: float=5.0, progress_step_interval: int=10, save_pickle_path: str='./cache_metrics.pkl', save_csv_path: str='./cache_metrics.csv', lines_per_subplot: int=6):
    param_points = set()
    for rate in rate_grid:
        for ron in r_over_n_grid:
            param_points.add((int(d2_fixed), int(d3_fixed), float(ron), float(rate)))
    for d3 in d3_grid:
        for d2 in d2_grid:
            param_points.add((int(d2), int(d3), float(r_over_n_fixed), float(rate_fixed)))
    param_points = sorted(param_points)
    total = len(param_points)
    print(f'[INFO] Unique parameter points = {total}')
    print(f'[INFO] SAN repeats per point = {repeats}')
    print(f'[INFO] Baseline rate = {baseline_rate}')
    pp = ProgressPrinter(total, time_interval=progress_time_interval, step_interval=progress_step_interval)
    cache = {}
    for d2, d3, ron, rate in param_points:
        mean_metrics_cached(cache=cache, n_players=n_players, d2=d2, d3=d3, r_over_n=ron, rate=rate, repeats=repeats, baseline_rate=baseline_rate, update_rule=update_rule, base_seed=base_seed)
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

def plot_2x2_dev_from_csv(csv_path: str, *, d2_fixed=6, d3_fixed=6, r_over_n_fixed=0.75, rate_fixed=0.3, d2_grid=np.arange(1, 9), d3_grid=np.arange(1, 11), r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), lines_per_subplot=6, y_margin_ratio=0.06, out_path=None, show=True):
    df = _load_csv_unique(csv_path)
    if 'dev_mean' not in df.columns:
        raise ValueError(f"CSV missing column 'dev_mean'. Got columns: {list(df.columns)}")
    ron_actual = _nearest_value(df['r_over_n'].unique(), r_over_n_fixed)
    rate_actual = _nearest_value(df['rate'].unique(), rate_fixed)
    df_ab = df[(df['d2'] == int(d2_fixed)) & (df['d3'] == int(d3_fixed))].copy()
    if df_ab.empty:
        raise ValueError(f'No rows for fixed (d2={d2_fixed}, d3={d3_fixed}).')
    A = {float(rate): [] for rate in rate_grid}
    for rate in rate_grid:
        for ron in r_over_n_grid:
            sub = df_ab[np.isclose(df_ab['rate'], float(rate)) & np.isclose(df_ab['r_over_n'], float(ron))]
            A[float(rate)].append(float(sub['dev_mean'].iloc[0]) if not sub.empty else np.nan)
    B = {float(ron): [] for ron in r_over_n_grid}
    for ron in r_over_n_grid:
        for rate in rate_grid:
            sub = df_ab[np.isclose(df_ab['r_over_n'], float(ron)) & np.isclose(df_ab['rate'], float(rate))]
            B[float(ron)].append(float(sub['dev_mean'].iloc[0]) if not sub.empty else np.nan)
    df_cd = df[np.isclose(df['r_over_n'], ron_actual) & np.isclose(df['rate'], rate_actual)].copy()
    if df_cd.empty:
        raise ValueError(f'No rows for (r_over_n≈{r_over_n_fixed}, rate≈{rate_fixed}). Nearest in CSV is (r_over_n={ron_actual}, rate={rate_actual}).')
    C = {int(d3): [] for d3 in d3_grid}
    for d3 in d3_grid:
        for d2 in d2_grid:
            sub = df_cd[(df_cd['d2'] == int(d2)) & (df_cd['d3'] == int(d3))]
            C[int(d3)].append(float(sub['dev_mean'].iloc[0]) if not sub.empty else np.nan)
    D = {int(d2): [] for d2 in d2_grid}
    for d2 in d2_grid:
        for d3 in d3_grid:
            sub = df_cd[(df_cd['d2'] == int(d2)) & (df_cd['d3'] == int(d3))]
            D[int(d2)].append(float(sub['dev_mean'].iloc[0]) if not sub.empty else np.nan)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    ax1, ax2, ax3, ax4 = axs.ravel()
    pickA = _pick_six_keys(sorted(A.keys()))[:lines_per_subplot]
    pickB = _pick_six_keys(sorted(B.keys()))[:lines_per_subplot]
    pickC = _pick_six_keys(sorted(C.keys()))[:lines_per_subplot]
    pickD = _pick_six_keys(sorted(D.keys()))[:lines_per_subplot]
    for rate in pickA:
        ax1.plot(r_over_n_grid, A[rate], marker='o', linewidth=1.2, label=f'$R$={rate:.2f}')
    ax1.set_title(f'(a) $d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax1.set_xlabel('$m$')
    ax1.set_ylabel('Deviate score')
    _set_tight_ylim(ax1, [y for k in pickA for y in A[k]], margin_ratio=y_margin_ratio)
    ax1.legend(fontsize=8, frameon=False, ncol=1)
    for ron in pickB:
        ax2.plot(rate_grid, B[ron], marker='o', linewidth=1.2, label=f'$m$={ron:.2f}')
    ax2.set_title(f'(b) $d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax2.set_xlabel('$R$')
    ax2.set_ylabel('Deviate score')
    _set_tight_ylim(ax2, [y for k in pickB for y in B[k]], margin_ratio=y_margin_ratio)
    ax2.legend(fontsize=8, frameon=False, ncol=1)
    for d3 in pickC:
        ax3.plot(d2_grid, C[d3], marker='o', linewidth=1.2, label=f'$d_3$={d3}')
    ax3.set_title(f'(c) $m={ron_actual:.3g},\\ R={rate_actual:.3g}$')
    ax3.set_xlabel('$d_2$')
    ax3.set_ylabel('Deviate score')
    _set_tight_ylim(ax3, [y for k in pickC for y in C[k]], margin_ratio=y_margin_ratio)
    ax3.legend(fontsize=8, frameon=False, ncol=1)
    for d2 in pickD:
        ax4.plot(d3_grid, D[d2], marker='o', linewidth=1.2, label=f'$d_2$={d2}')
    ax4.set_title(f'(d) $m={ron_actual:.3g},\\ R={rate_actual:.3g}$')
    ax4.set_xlabel('$d_3$')
    ax4.set_ylabel('Deviate score')
    _set_tight_ylim(ax4, [y for k in pickD for y in D[k]], margin_ratio=y_margin_ratio)
    ax4.legend(fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (fig, {'A': A, 'B': B, 'C': C, 'D': D})

def plot_B_dev_from_csv(csv_path: str, *, d2_fixed=6, d3_fixed=6, r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), lines_per_subplot=6, y_margin_ratio=0.06, out_path=None, show=True):
    df = _load_csv_unique(csv_path)
    if 'dev_mean' not in df.columns:
        raise ValueError(f"CSV missing column 'dev_mean'. Got columns: {list(df.columns)}")
    df_ab = df[(df['d2'] == int(d2_fixed)) & (df['d3'] == int(d3_fixed))].copy()
    if df_ab.empty:
        raise ValueError(f'No rows for fixed (d2={d2_fixed}, d3={d3_fixed}).')
    B = {float(ron): [] for ron in r_over_n_grid}
    for ron in r_over_n_grid:
        for rate in rate_grid:
            sub = df_ab[np.isclose(df_ab['r_over_n'], float(ron)) & np.isclose(df_ab['rate'], float(rate))]
            B[float(ron)].append(float(sub['dev_mean'].iloc[0]) if not sub.empty else np.nan)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    pickB = _pick_six_keys(sorted(B.keys()))[:lines_per_subplot]
    for ron in pickB:
        ax.plot(rate_grid, B[ron], marker='o', linewidth=1.2, label=f'$m$={ron:.2f}')
    ax.set_title(f'$d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax.set_xlabel('$R$')
    ax.set_ylabel('Deviate score')
    _set_tight_ylim(ax, [y for k in pickB for y in B[k]], margin_ratio=y_margin_ratio)
    ax.legend(fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (fig, {'B': B})

def plot_2x2_steps_from_csv(csv_path: str, *, d2_fixed=6, d3_fixed=6, r_over_n_fixed=0.75, rate_fixed=0.3, d2_grid=np.arange(1, 9), d3_grid=np.arange(1, 11), r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), lines_per_subplot=6, y_margin_ratio=0.06, out_path=None, show=True):
    df = _load_csv_unique(csv_path)
    if 'steps_mean' not in df.columns:
        raise ValueError(f"CSV missing column 'steps_mean'. Got columns: {list(df.columns)}")
    ron_actual = _nearest_value(df['r_over_n'].unique(), r_over_n_fixed)
    rate_actual = _nearest_value(df['rate'].unique(), rate_fixed)
    df_ab = df[(df['d2'] == int(d2_fixed)) & (df['d3'] == int(d3_fixed))].copy()
    if df_ab.empty:
        raise ValueError(f'No rows for fixed (d2={d2_fixed}, d3={d3_fixed}).')
    A = {float(rate): [] for rate in rate_grid}
    for rate in rate_grid:
        for ron in r_over_n_grid:
            sub = df_ab[np.isclose(df_ab['rate'], float(rate)) & np.isclose(df_ab['r_over_n'], float(ron))]
            A[float(rate)].append(float(sub['steps_mean'].iloc[0]) if not sub.empty else np.nan)
    B = {float(ron): [] for ron in r_over_n_grid}
    for ron in r_over_n_grid:
        for rate in rate_grid:
            sub = df_ab[np.isclose(df_ab['r_over_n'], float(ron)) & np.isclose(df_ab['rate'], float(rate))]
            B[float(ron)].append(float(sub['steps_mean'].iloc[0]) if not sub.empty else np.nan)
    df_cd = df[np.isclose(df['r_over_n'], ron_actual) & np.isclose(df['rate'], rate_actual)].copy()
    if df_cd.empty:
        raise ValueError(f'No rows for (r_over_n≈{r_over_n_fixed}, rate≈{rate_fixed}). Nearest in CSV is (r_over_n={ron_actual}, rate={rate_actual}).')
    C = {int(d3): [] for d3 in d3_grid}
    for d3 in d3_grid:
        for d2 in d2_grid:
            sub = df_cd[(df_cd['d2'] == int(d2)) & (df_cd['d3'] == int(d3))]
            C[int(d3)].append(float(sub['steps_mean'].iloc[0]) if not sub.empty else np.nan)
    D = {int(d2): [] for d2 in d2_grid}
    for d2 in d2_grid:
        for d3 in d3_grid:
            sub = df_cd[(df_cd['d2'] == int(d2)) & (df_cd['d3'] == int(d3))]
            D[int(d2)].append(float(sub['steps_mean'].iloc[0]) if not sub.empty else np.nan)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    ax1, ax2, ax3, ax4 = axs.ravel()
    pickA = _pick_six_keys(sorted(A.keys()))[:lines_per_subplot]
    pickB = _pick_six_keys(sorted(B.keys()))[:lines_per_subplot]
    pickC = _pick_six_keys(sorted(C.keys()))[:lines_per_subplot]
    pickD = _pick_six_keys(sorted(D.keys()))[:lines_per_subplot]
    for rate in pickA:
        ax1.plot(r_over_n_grid, A[rate], marker='o', linewidth=1.2, label=f'$R$={rate:.2f}')
    ax1.set_title(f'(a) $d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax1.set_xlabel('$m$')
    ax1.set_ylabel('Mean convergent steps')
    _set_tight_ylim(ax1, [y for k in pickA for y in A[k]], margin_ratio=y_margin_ratio)
    ax1.legend(fontsize=8, frameon=False, ncol=1)
    for ron in pickB:
        ax2.plot(rate_grid, B[ron], marker='o', linewidth=1.2, label=f'$m$={ron:.2f}')
    ax2.set_title(f'(b) $d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax2.set_xlabel('$R$')
    ax2.set_ylabel('Mean convergent steps')
    _set_tight_ylim(ax2, [y for k in pickB for y in B[k]], margin_ratio=y_margin_ratio)
    ax2.legend(fontsize=8, frameon=False, ncol=1)
    for d3 in pickC:
        ax3.plot(d2_grid, C[d3], marker='o', linewidth=1.2, label=f'$d_3$={d3}')
    ax3.set_title(f'(c) $m={ron_actual:.3g},\\ R={rate_actual:.3g}$')
    ax3.set_xlabel('$d_2$')
    ax3.set_ylabel('Mean convergent steps')
    _set_tight_ylim(ax3, [y for k in pickC for y in C[k]], margin_ratio=y_margin_ratio)
    ax3.legend(fontsize=8, frameon=False, ncol=1)
    for d2 in pickD:
        ax4.plot(d3_grid, D[d2], marker='o', linewidth=1.2, label=f'$d_2$={d2}')
    ax4.set_title(f'(d) $m={ron_actual:.3g},\\ R={rate_actual:.3g}$')
    ax4.set_xlabel('$d_3$')
    ax4.set_ylabel('Mean convergent steps')
    _set_tight_ylim(ax4, [y for k in pickD for y in D[k]], margin_ratio=y_margin_ratio)
    ax4.legend(fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (fig, {'A': A, 'B': B, 'C': C, 'D': D})

def plot_B_steps_from_csv(csv_path: str, *, d2_fixed=6, d3_fixed=6, r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), lines_per_subplot=6, y_margin_ratio=0.06, out_path=None, show=True):
    df = _load_csv_unique(csv_path)
    if 'steps_mean' not in df.columns:
        raise ValueError(f"CSV missing column 'steps_mean'. Got columns: {list(df.columns)}")
    df_ab = df[(df['d2'] == int(d2_fixed)) & (df['d3'] == int(d3_fixed))].copy()
    if df_ab.empty:
        raise ValueError(f'No rows for fixed (d2={d2_fixed}, d3={d3_fixed}).')
    B = {float(ron): [] for ron in r_over_n_grid}
    for ron in r_over_n_grid:
        for rate in rate_grid:
            sub = df_ab[np.isclose(df_ab['r_over_n'], float(ron)) & np.isclose(df_ab['rate'], float(rate))]
            B[float(ron)].append(float(sub['steps_mean'].iloc[0]) if not sub.empty else np.nan)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    pickB = _pick_six_keys(sorted(B.keys()))[:lines_per_subplot]
    for ron in pickB:
        ax.plot(rate_grid, B[ron], marker='o', linewidth=1.2, label=f'$m$={ron:.2f}')
    ax.set_title(f'$d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax.set_xlabel('$R$')
    ax.set_ylabel('Mean convergent steps')
    _set_tight_ylim(ax, [y for k in pickB for y in B[k]], margin_ratio=y_margin_ratio)
    ax.legend(fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (fig, {'B': B})

def plot_B_dev_steps_together(csv_path: str, *, d2_fixed=6, d3_fixed=6, r_over_n_grid=np.linspace(0.5, 1.5, 21), rate_grid=np.linspace(0.1, 1.0, 10), lines_per_subplot=6, y_margin_ratio=0.06, out_path=None, show=True):
    """
    Put the two 'panel (b)' plots together in one figure (1×2 layout):

      Left : Deviate score
      Right: Mean convergent steps

    Both panels:
      x = R
      series over m (= r_over_n)
      fixed d2, d3
    """
    df = _load_csv_unique(csv_path)
    required = {'dev_mean', 'steps_mean'}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'CSV missing columns: {missing}. Got: {list(df.columns)}')
    df_ab = df[(df['d2'] == int(d2_fixed)) & (df['d3'] == int(d3_fixed))].copy()
    if df_ab.empty:
        raise ValueError(f'No rows for fixed (d2={d2_fixed}, d3={d3_fixed}).')
    B_dev = {float(ron): [] for ron in r_over_n_grid}
    B_steps = {float(ron): [] for ron in r_over_n_grid}
    for ron in r_over_n_grid:
        for rate in rate_grid:
            sub = df_ab[np.isclose(df_ab['r_over_n'], float(ron)) & np.isclose(df_ab['rate'], float(rate))]
            if not sub.empty:
                B_dev[float(ron)].append(float(sub['dev_mean'].iloc[0]))
                B_steps[float(ron)].append(float(sub['steps_mean'].iloc[0]))
            else:
                B_dev[float(ron)].append(np.nan)
                B_steps[float(ron)].append(np.nan)
    pickB = _pick_six_keys(sorted(B_dev.keys()))[:lines_per_subplot]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ron in pickB:
        ax1.plot(rate_grid, B_dev[ron], marker='o', label=f'$m$={ron:.2f}')
    ax1.set_title(f'(a) $d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax1.set_xlabel('$R$')
    ax1.set_ylabel('Deviate score')
    _set_tight_ylim(ax1, [y for k in pickB for y in B_dev[k]], margin_ratio=y_margin_ratio)
    ax1.legend(fontsize=8, frameon=False, ncol=1)
    for ron in pickB:
        ax2.plot(rate_grid, B_steps[ron], marker='o', label=f'$m$={ron:.2f}')
    ax2.set_title(f'(b) $d_2={d2_fixed},\\ d_3={d3_fixed}$')
    ax2.set_xlabel('$R$')
    ax2.set_ylabel('Mean convergent steps')
    _set_tight_ylim(ax2, [y for k in pickB for y in B_steps[k]], margin_ratio=y_margin_ratio)
    ax2.legend(fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (fig, {'B_dev': B_dev, 'B_steps': B_steps})
