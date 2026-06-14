#!/usr/bin/env python3
"""
discrimination_figures.py

Figures for the signal-versus-noise (reliability) analysis:
  fig14 — single-game reliability (ICC) vs skill, plus the variance
          decomposition showing that traditional scoring's larger elite-level
          spread is mostly within-player noise, not between-player signal.
  fig15 — streakiness discrimination at fixed marginal skill: traditional
          scoring tracks sequencing ability; World Bowling is blind to it.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from discrimination_analysis import (
    reliability_vs_skill, streakiness_discrimination,
)
from plot_style import (
    TRAD_COLOR, WORLD_COLOR, TRAD_LINE, WORLD_LINE, TRAD_LABEL, WORLD_LABEL,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'{name}.{ext}')
        fig.savefig(path)
        print(f'  Saved: {path}')
    plt.close(fig)


def fig_reliability(reliability_rows):
    s = np.array([r['strike_rate'] for r in reliability_rows]) * 100
    trad_icc = np.array([r['trad_icc'] for r in reliability_rows])
    world_icc = np.array([r['world_icc'] for r in reliability_rows])
    trad_noise = np.sqrt([r['trad_sigma_w'] for r in reliability_rows])
    world_noise = np.sqrt([r['world_sigma_w'] for r in reliability_rows])
    trad_sig = np.sqrt([r['trad_sigma_b'] for r in reliability_rows])
    world_sig = np.sqrt([r['world_sigma_b'] for r in reliability_rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: single-game reliability (ICC) — the two systems track each other.
    ax1.plot(s, trad_icc, label=TRAD_LABEL, **TRAD_LINE, marker='o', markersize=5)
    ax1.plot(s, world_icc, label=WORLD_LABEL, **WORLD_LINE, marker='s', markersize=5)
    ax1.set_xlabel('Strike Rate (%)')
    ax1.set_ylabel('Single-game reliability (ICC)')
    ax1.set_title('Reliability: signal / (signal + noise)')
    ax1.set_ylim(0, max(trad_icc.max(), world_icc.max()) * 1.25)
    ax1.legend()

    # Right: the variance decomposition. Noise (within-player SD) diverges at
    # elite level while signal (between-player SD) stays comparable, so the extra
    # spread under traditional scoring is mostly noise.
    ax2.plot(s, trad_noise, color=TRAD_COLOR, linestyle='-', linewidth=1.5,
             marker='o', markersize=4, label='Traditional: noise (within)')
    ax2.plot(s, world_noise, color=WORLD_COLOR, linestyle='--', linewidth=1.5,
             marker='s', markersize=4, label='World Bowling: noise (within)')
    ax2.plot(s, trad_sig, color=TRAD_COLOR, linestyle=':', linewidth=1.5,
             marker='^', markersize=4, label='Traditional: signal (between)')
    ax2.plot(s, world_sig, color=WORLD_COLOR, linestyle=(0, (1, 1)), linewidth=1.5,
             marker='v', markersize=4, label='World Bowling: signal (between)')
    ax2.set_xlabel('Strike Rate (%)')
    ax2.set_ylabel('Score SD component')
    ax2.set_title('Variance decomposition: noise vs signal')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, 'fig14_reliability')


def fig_streakiness(details, pair, summary):
    phis = details['phis']
    trad_means = details['trad_player_means']
    world_means = details['world_player_means']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: player mean score vs true streakiness. Traditional rises; WB is flat.
    ax1.scatter(phis, trad_means, s=14, color=TRAD_COLOR, alpha=0.6,
                label=TRAD_LABEL)
    ax1.scatter(phis, world_means, s=14, color=WORLD_COLOR, alpha=0.6,
                marker='s', label=WORLD_LABEL)
    for means, color, ls in [(trad_means, TRAD_COLOR, '-'),
                             (world_means, WORLD_COLOR, '--')]:
        coef = np.polyfit(phis, means, 1)
        xs = np.linspace(phis.min(), phis.max(), 50)
        ax1.plot(xs, np.polyval(coef, xs), color=color, linestyle=ls, linewidth=1.5)
    ax1.set_xlabel('Streakiness (lag-1 strike autocorrelation $\\phi$)')
    ax1.set_ylabel('Player mean score')
    ax1.set_title('Same marginal skill, different streakiness\n'
                  f'(Trad. $r={summary["trad_corr_phi"]:+.2f}$, '
                  f'WB $r={summary["world_corr_phi"]:+.2f}$)')
    ax1.legend()

    # Right: head-to-head game-score distributions, low vs high streak.
    bins = np.arange(150, 305, 5)
    ax2.hist(pair['_t_lo'], bins=bins, density=True, alpha=0.45, color=TRAD_COLOR,
             label=f'Trad., $\\phi$=0')
    ax2.hist(pair['_t_hi'], bins=bins, density=True, alpha=0.45, color=TRAD_COLOR,
             histtype='step', linewidth=1.8,
             label=f'Trad., $\\phi$={pair["phi_high"]}')
    ax2.hist(pair['_w_lo'], bins=bins, density=True, alpha=0.35, color=WORLD_COLOR,
             label=f'WB, $\\phi$=0')
    ax2.hist(pair['_w_hi'], bins=bins, density=True, alpha=0.35, color=WORLD_COLOR,
             histtype='step', linewidth=1.8, linestyle='--',
             label=f'WB, $\\phi$={pair["phi_high"]}')
    ax2.axvline(pair['trad_mean_low'], color=TRAD_COLOR, linewidth=1, alpha=0.7)
    ax2.axvline(pair['trad_mean_high'], color=TRAD_COLOR, linewidth=1, alpha=0.7,
                linestyle=':')
    ax2.set_xlabel('Game score')
    ax2.set_ylabel('Density')
    ax2.set_title('Low vs high streak at fixed skill\n'
                  f'(Trad. $d={pair["trad_d"]:+.2f}$, WB $d={pair["world_d"]:+.2f}$)')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, 'fig15_streakiness')


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print('Generating discrimination figures...')
    reliability_rows = reliability_vs_skill()
    summary, pair, details = streakiness_discrimination()
    fig_reliability(reliability_rows)
    fig_streakiness(details, pair, summary)
    print('Done — discrimination figures saved.')


if __name__ == '__main__':
    main()
