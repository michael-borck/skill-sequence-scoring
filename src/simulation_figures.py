#!/usr/bin/env python3
"""
simulation_figures.py

Generate figures for the simulation (Part 2) section of the paper.
"""

import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from bowling_simulation import (
    SKILL_TIERS, simulate_games, crossover_analysis, score_separation
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

TRAD_COLOR = '#2166ac'
WORLD_COLOR = '#b2182b'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def fig_tier_distributions():
    """
    Score distributions at each skill tier, traditional vs World Bowling.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (name, params) in enumerate(SKILL_TIERS.items()):
        ax = axes[i]
        trad, world = simulate_games(
            30_000, params['p_strike'], params['p_spare'], params['pin_mean']
        )

        bins = np.arange(0, 305, 5)
        ax.hist(trad, bins=bins, alpha=0.6, color=TRAD_COLOR, label='Traditional',
                density=True)
        ax.hist(world, bins=bins, alpha=0.6, color=WORLD_COLOR, label='World Bowling',
                density=True)
        ax.set_title(f'{name}\n(strike rate: {params["p_strike"]*100:.0f}%)')
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=8)

        # Annotate means
        ax.axvline(np.mean(trad), color=TRAD_COLOR, linestyle='--', alpha=0.6)
        ax.axvline(np.mean(world), color=WORLD_COLOR, linestyle='--', alpha=0.6)

    # Hide unused subplot
    axes[5].set_visible(False)

    fig.suptitle('Score Distributions by Skill Tier', fontsize=14, y=1.02)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'fig8_tier_distributions.{ext}')
        fig.savefig(path)
        print(f'  Saved: {path}')
    plt.close(fig)


def fig_mean_scores_by_tier():
    """
    Mean scores under each system across skill tiers.
    """
    tier_names = list(SKILL_TIERS.keys())
    trad_means = []
    world_means = []
    trad_stds = []
    world_stds = []

    for name, params in SKILL_TIERS.items():
        trad, world = simulate_games(
            30_000, params['p_strike'], params['p_spare'], params['pin_mean']
        )
        trad_means.append(np.mean(trad))
        world_means.append(np.mean(world))
        trad_stds.append(np.std(trad))
        world_stds.append(np.std(world))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(tier_names))

    # Left: mean scores
    ax1.bar(x - 0.18, trad_means, 0.35, color=TRAD_COLOR, label='Traditional', alpha=0.85)
    ax1.bar(x + 0.18, world_means, 0.35, color=WORLD_COLOR, label='World Bowling', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_names, fontsize=9)
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Mean Score by Skill Tier')
    ax1.legend()
    for i, (t, w) in enumerate(zip(trad_means, world_means)):
        ax1.text(i - 0.18, t + 2, f'{t:.0f}', ha='center', fontsize=7, color=TRAD_COLOR)
        ax1.text(i + 0.18, w + 2, f'{w:.0f}', ha='center', fontsize=7, color=WORLD_COLOR)

    # Right: standard deviations (score spread)
    ax2.plot(tier_names, trad_stds, 'o-', color=TRAD_COLOR, label='Traditional', markersize=7)
    ax2.plot(tier_names, world_stds, 's-', color=WORLD_COLOR, label='World Bowling', markersize=7)
    ax2.set_ylabel('Score Standard Deviation')
    ax2.set_title('Score Spread by Skill Tier')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'fig9_mean_scores_tiers.{ext}')
        fig.savefig(path)
        print(f'  Saved: {path}')
    plt.close(fig)

    return trad_stds, world_stds


def fig_crossover():
    """
    Discrimination ratio and score statistics across the full strike rate sweep.
    """
    # Load from CSV if available, otherwise recompute
    csv_path = os.path.join(DATA_DIR, 'simulation_crossover.csv')
    strike_rates, trad_means, world_means = [], [], []
    trad_stds, world_stds, disc_ratios = [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            strike_rates.append(float(row['strike_rate']) * 100)
            trad_means.append(float(row['trad_mean']))
            world_means.append(float(row['world_mean']))
            trad_stds.append(float(row['trad_stdev']))
            world_stds.append(float(row['world_stdev']))
            disc_ratios.append(float(row['discrimination_ratio']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Mean scores diverge
    ax1.plot(strike_rates, trad_means, '-', color=TRAD_COLOR, label='Traditional', linewidth=1.5)
    ax1.plot(strike_rates, world_means, '-', color=WORLD_COLOR, label='World Bowling', linewidth=1.5)
    ax1.fill_between(strike_rates,
                     [m - s for m, s in zip(trad_means, trad_stds)],
                     [m + s for m, s in zip(trad_means, trad_stds)],
                     color=TRAD_COLOR, alpha=0.15)
    ax1.fill_between(strike_rates,
                     [m - s for m, s in zip(world_means, world_stds)],
                     [m + s for m, s in zip(world_means, world_stds)],
                     color=WORLD_COLOR, alpha=0.15)
    ax1.set_xlabel('Strike Rate (%)')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Mean Score ± 1 SD by Strike Rate')
    ax1.legend()

    # Right: Standard deviations cross
    ax2.plot(strike_rates, trad_stds, '-', color=TRAD_COLOR, label='Traditional SD', linewidth=1.5)
    ax2.plot(strike_rates, world_stds, '-', color=WORLD_COLOR, label='World Bowling SD', linewidth=1.5)
    ax2.set_xlabel('Strike Rate (%)')
    ax2.set_ylabel('Score Standard Deviation')
    ax2.set_title('Score Spread by Strike Rate')
    ax2.legend()

    # Find and mark the crossover point where SDs cross
    for i in range(1, len(trad_stds)):
        if trad_stds[i] > world_stds[i] and trad_stds[i-1] <= world_stds[i-1]:
            cross_x = strike_rates[i]
            cross_y = trad_stds[i]
            ax2.axvline(cross_x, color='gray', linestyle=':', alpha=0.5)
            ax2.annotate(f'Crossover ≈ {cross_x:.0f}%',
                         xy=(cross_x, cross_y),
                         xytext=(cross_x + 5, cross_y + 3),
                         fontsize=9, style='italic',
                         arrowprops=dict(arrowstyle='->', color='gray'))
            break

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'fig10_crossover_analysis.{ext}')
        fig.savefig(path)
        print(f'  Saved: {path}')
    plt.close(fig)


def fig_professional_sequences():
    """
    Side-by-side comparison of specific professional-level sequences.
    """
    from bowling_analysis import score_traditional, score_world

    sequences = [
        ('12 strikes\n(perfect)', [10]*12),
        ('10X + spare\n(7,3) + 7', [10]*10 + [7, 3, 7]),
        ('8X + 2 spare\n(7,3) + 7', [10]*8 + [7, 3] + [7, 3, 7]),
        ('Spares then\n4X + X,X', [5,5]*6 + [10]*4 + [10, 10]),
        ('4X then\nspares + 5', [10]*4 + [5,5]*6 + [5]),
        ('Alt X/spare\n×5', [10, 5, 5]*3 + [10, 5, 5, 10, 5, 5, 10, 5, 5, 5]),
        ('All spares\n(5,5) + 5', [5, 5]*10 + [5]),
    ]

    names = [s[0] for s in sequences]
    trad = [score_traditional(s[1]) for s in sequences]
    world = [score_world(s[1]) for s in sequences]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, trad, width, color=TRAD_COLOR,
                   label='Traditional', alpha=0.85)
    bars2 = ax.bar(x + width/2, world, width, color=WORLD_COLOR,
                   label='World Bowling', alpha=0.85)

    ax.set_ylabel('Game Score')
    ax.set_title('Professional-Level Sequences: Traditional vs World Bowling')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom',
                fontsize=7, color=TRAD_COLOR)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{int(bar.get_height())}', ha='center', va='bottom',
                    fontsize=7, color=WORLD_COLOR)

    ax.set_ylim(0, 330)

    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'fig11_professional_sequences.{ext}')
        fig.savefig(path)
        print(f'  Saved: {path}')
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('Generating simulation figures...')
    fig_tier_distributions()
    fig_mean_scores_by_tier()
    fig_crossover()
    fig_professional_sequences()
    print('Done — all simulation figures saved.')


if __name__ == '__main__':
    main()
