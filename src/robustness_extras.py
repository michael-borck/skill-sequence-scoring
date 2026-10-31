#!/usr/bin/env python3
"""
robustness_extras.py

Adds the uncertainty and sensitivity results requested after the first audit of
the manuscript:

  1. Moments for a player at the PBA-average strike rate (the mixture mean that
     reproduces McCarthy's published aggregate moments).
  2. Score-spread gaps for every skill tier under all three simulation models.
  3. Bootstrap confidence intervals for the streakiness correlations (r) and the
     extreme-group effect size (Cohen's d).
  4. Win probability over a 48-game series as a function of the streakiness gap,
     so the headline "90% tournament" figure can be read against any empirical
     population spread of streakiness.

All results are written to ../data/ as CSV.

Usage:  cd src && python robustness_extras.py
"""

import csv
import os

import numpy as np

from bowling_analysis import score_traditional, score_world
from bowling_simulation import (
    SKILL_TIERS,
    simulate_games,
    simulate_games_momentum,
    simulate_games_markov,
)
from discrimination_analysis import (
    cohens_d,
    icc_oneway,
    skill_params,
    _simulate_player,
    streakiness_discrimination,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PBA_MEAN_STRIKE = 0.54


def _save(rows, filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {path}')
    return path


def pba_average_moments(n_games=50_000, seed=20240701):
    """Moments for a single player at the PBA-average strike rate."""
    p_spare, pin_mean = skill_params(PBA_MEAN_STRIKE)
    t, w = simulate_games(n_games, PBA_MEAN_STRIKE, p_spare, pin_mean, seed=seed)
    row = {
        'p_strike': PBA_MEAN_STRIKE,
        'trad_mean': float(np.mean(t)), 'trad_sd': float(np.std(t)),
        'world_mean': float(np.mean(w)), 'world_sd': float(np.std(w)),
        'n_games': int(len(t)),
    }
    print(f'\nPBA-average player ({PBA_MEAN_STRIKE:.0%} strike): '
          f'traditional {row["trad_mean"]:.1f} (SD {row["trad_sd"]:.1f}), '
          f'World Bowling {row["world_mean"]:.1f} (SD {row["world_sd"]:.1f})')
    return row


def robustness_gaps(n_games=30_000):
    """Score-spread gap (traditional minus World Bowling SD) by tier and model."""
    models = {
        'Independent': simulate_games,
        'Momentum': simulate_games_momentum,
        'Markov': simulate_games_markov,
    }
    rows = []
    for tier, params in SKILL_TIERS.items():
        for model, fn in models.items():
            t, w = fn(n_games, params['p_strike'], params['p_spare'],
                      params['pin_mean'])
            rows.append({
                'tier': tier,
                'p_strike': params['p_strike'],
                'model': model,
                'trad_sd': float(np.std(t)), 'world_sd': float(np.std(w)),
                'sd_gap': float(np.std(t) - np.std(w)),
            })
    print('\nSD gaps by tier and model:')
    for r in rows:
        print(f'  {r["tier"]:<13} {r["model"]:<12} '
              f'trad {r["trad_sd"]:5.2f}  WB {r["world_sd"]:5.2f}  '
              f'gap {r["sd_gap"]:+5.2f}')
    return rows


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def streakiness_cis(n_boot=2000, seed=20240702):
    """Bootstrap CIs for the streakiness correlations and the extreme-group d."""
    summary, pair, details = streakiness_discrimination()
    rng = np.random.default_rng(seed)

    phis = np.asarray(details['phis'])
    trad_means = np.asarray(details['trad_player_means'])
    world_means = np.asarray(details['world_player_means'])
    n_players = len(phis)

    r_trad, r_world, d_trad, d_world = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_players, n_players)
        r_trad.append(_pearson(phis[idx], trad_means[idx]))
        r_world.append(_pearson(phis[idx], world_means[idx]))

        gi = rng.integers(0, len(pair['_t_lo']), len(pair['_t_lo']))
        d_trad.append(cohens_d(pair['_t_hi'][gi], pair['_t_lo'][gi]))
        d_world.append(cohens_d(pair['_w_hi'][gi], pair['_w_lo'][gi]))

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    row = {
        'r_trad': summary['trad_corr_phi'],
        'r_trad_lo': ci(r_trad)[0], 'r_trad_hi': ci(r_trad)[1],
        'r_world': summary['world_corr_phi'],
        'r_world_lo': ci(r_world)[0], 'r_world_hi': ci(r_world)[1],
        'd_trad': pair['trad_d'],
        'd_trad_lo': ci(d_trad)[0], 'd_trad_hi': ci(d_trad)[1],
        'd_world': pair['world_d'],
        'd_world_lo': ci(d_world)[0], 'd_world_hi': ci(d_world)[1],
        'n_boot': n_boot,
    }
    print(f'\nStreakiness r (traditional): {row["r_trad"]:+.3f} '
          f'[{row["r_trad_lo"]:+.3f}, {row["r_trad_hi"]:+.3f}]')
    print(f'Streakiness r (World Bowling): {row["r_world"]:+.3f} '
          f'[{row["r_world_lo"]:+.3f}, {row["r_world_hi"]:+.3f}]')
    print(f'Cohen d (traditional): {row["d_trad"]:+.3f} '
          f'[{row["d_trad_lo"]:+.3f}, {row["d_trad_hi"]:+.3f}]')
    print(f'Cohen d (World Bowling): {row["d_world"]:+.3f} '
          f'[{row["d_world_lo"]:+.3f}, {row["d_world_hi"]:+.3f}]')
    return row


def series_gap_sweep(p_strike=0.66, deltas=(0.0, 0.05, 0.10, 0.15, 0.20,
                                            0.25, 0.30, 0.35),
                     n_games=40_000, n_series=30_000, series_length=48,
                     seed=20240606):
    """
    48-game win probability as a function of the streakiness gap. Each gap uses
    the same seed and the same phi=0 reference pool, so the delta=0.35 row
    reproduces the main series_compounding() result exactly.
    """
    p_spare, pin_mean = skill_params(p_strike)
    rows = []
    for delta in deltas:
        rng = np.random.default_rng(seed)
        mc_rng = np.random.default_rng(seed + 1)
        t_lo, w_lo = _simulate_player(rng, p_strike, p_spare, pin_mean, n_games,
                                      model='autocorr', phi=0.0)
        if delta == 0.0:
            t_hi, w_hi = t_lo, w_lo
        else:
            t_hi, w_hi = _simulate_player(rng, p_strike, p_spare, pin_mean,
                                          n_games, model='autocorr', phi=delta)

        def winprob(pool_hi, pool_lo):
            hi = mc_rng.choice(pool_hi, size=(n_series, series_length)).sum(axis=1)
            lo = mc_rng.choice(pool_lo, size=(n_series, series_length)).sum(axis=1)
            p = float(np.mean(hi > lo) + 0.5 * np.mean(hi == lo))
            se = float(np.sqrt(p * (1 - p) / n_series))
            return p, se

        tp, tse = winprob(t_hi, t_lo)
        wp, wse = winprob(w_hi, w_lo)
        rows.append({
            'phi_gap': float(delta), 'series_length': series_length,
            'trad_winprob': tp, 'trad_se': tse,
            'world_winprob': wp, 'world_se': wse,
        })
        print(f'  phi gap {delta:.2f}: traditional {tp*100:.1f}% '
              f'(SE {tse*100:.1f}), World Bowling {wp*100:.1f}% '
              f'(SE {wse*100:.1f})')
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    _save([pba_average_moments()], 'pba_average_moments.csv')
    _save(robustness_gaps(), 'robustness_gaps.csv')
    _save([streakiness_cis()], 'streakiness_cis.csv')
    print('\n48-game series win probability by streakiness gap:')
    _save(series_gap_sweep(), 'series_gap_sweep.csv')


if __name__ == '__main__':
    main()
