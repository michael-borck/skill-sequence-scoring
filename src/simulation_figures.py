#!/usr/bin/env python3
"""
simulation_figures.py

Generate figures for the simulation (Part 2) section of the paper.
Uses B&W-friendly palette from plot_style.py.
"""

import os
import csv

import matplotlib.pyplot as plt
import numpy as np

from bowling_simulation import (
    SKILL_TIERS, simulate_games, simulate_games_momentum,
    simulate_games_markov, crossover_analysis, score_separation
)
from plot_style import (
    TRAD_COLOR, WORLD_COLOR, TRAD_FILL, WORLD_FILL,
    TRAD_LINE, WORLD_LINE, TRAD_BAR, WORLD_BAR,
    TRAD_LABEL, WORLD_LABEL,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'{name}.{ext}')
        fig.savefig(path)
        print(f'  Saved: {path}')
    plt.close(fig)


def fig_tier_distributions():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (name, params) in enumerate(SKILL_TIERS.items()):
        ax = axes[i]
        trad, world = simulate_games(
            30_000, params['p_strike'], params['p_spare'], params['pin_mean']
        )

        bins = np.arange(0, 305, 5)
        ax.hist(trad, bins=bins, alpha=0.7, color=TRAD_FILL,
                edgecolor=TRAD_COLOR, linewidth=0.5,
                label=TRAD_LABEL, density=True)
        ax.hist(world, bins=bins, alpha=0.5, color=WORLD_FILL,
                edgecolor=WORLD_COLOR, linewidth=0.5, hatch='///',
                label=WORLD_LABEL, density=True)
        ax.set_title(f'{name}\n(strike rate: {params["p_strike"]*100:.0f}%)')
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=8)

        ax.axvline(np.mean(trad), color=TRAD_COLOR, linestyle='-', alpha=0.6, linewidth=1)
        ax.axvline(np.mean(world), color=WORLD_COLOR, linestyle='--', alpha=0.6, linewidth=1)

    fig.suptitle('Score Distributions by Skill Tier', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig8_tier_distributions')


def fig_mean_scores_by_tier():
    tier_names = list(SKILL_TIERS.keys())
    trad_means, world_means = [], []
    trad_stds, world_stds = [], []

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

    ax1.bar(x - 0.18, trad_means, 0.35, label=TRAD_LABEL, **TRAD_BAR)
    ax1.bar(x + 0.18, world_means, 0.35, label=WORLD_LABEL, **WORLD_BAR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_names, fontsize=9)
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Mean Score by Skill Tier')
    ax1.legend()
    for i, (t, w) in enumerate(zip(trad_means, world_means)):
        ax1.text(i - 0.18, t + 2, f'{t:.0f}', ha='center', fontsize=7, color=TRAD_COLOR)
        ax1.text(i + 0.18, w + 2, f'{w:.0f}', ha='center', fontsize=7, color=WORLD_COLOR)

    ax2.plot(tier_names, trad_stds, '-', marker='o', color=TRAD_COLOR,
             label=TRAD_LABEL, markersize=7)
    ax2.plot(tier_names, world_stds, '--', marker='s', color=WORLD_COLOR,
             label=WORLD_LABEL, markersize=7)
    ax2.set_ylabel('Score Standard Deviation')
    ax2.set_title('Score Spread by Skill Tier')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    save_fig(fig, 'fig9_mean_scores_tiers')
    return trad_stds, world_stds


def fig_crossover():
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

    # Left: Mean scores ± 1 SD
    ax1.plot(strike_rates, trad_means, label=TRAD_LABEL, **TRAD_LINE)
    ax1.plot(strike_rates, world_means, label=WORLD_LABEL, **WORLD_LINE)
    ax1.fill_between(strike_rates,
                     [m - s for m, s in zip(trad_means, trad_stds)],
                     [m + s for m, s in zip(trad_means, trad_stds)],
                     color=TRAD_COLOR, alpha=0.1)
    ax1.fill_between(strike_rates,
                     [m - s for m, s in zip(world_means, world_stds)],
                     [m + s for m, s in zip(world_means, world_stds)],
                     color=WORLD_COLOR, alpha=0.08, hatch='...')
    ax1.set_xlabel('Strike Rate (%)')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Mean Score ± 1 SD by Strike Rate')
    ax1.legend()

    # Right: Standard deviations cross
    ax2.plot(strike_rates, trad_stds, label=TRAD_LABEL + ' SD', **TRAD_LINE)
    ax2.plot(strike_rates, world_stds, label=WORLD_LABEL + ' SD', **WORLD_LINE)
    ax2.set_xlabel('Strike Rate (%)')
    ax2.set_ylabel('Score Standard Deviation')
    ax2.set_title('Score Spread by Strike Rate')
    ax2.legend()

    for i in range(1, len(trad_stds)):
        if trad_stds[i] > world_stds[i] and trad_stds[i-1] <= world_stds[i-1]:
            # Linearly interpolate the exact crossing of the two SD curves.
            d0 = trad_stds[i-1] - world_stds[i-1]
            d1 = trad_stds[i] - world_stds[i]
            frac = (-d0) / (d1 - d0) if (d1 - d0) != 0 else 0.0
            cross_x = strike_rates[i-1] + frac * (strike_rates[i] - strike_rates[i-1])
            cross_y = world_stds[i-1] + frac * (world_stds[i] - world_stds[i-1])
            ax2.axvline(cross_x, color='gray', linestyle=':', alpha=0.5)
            ax2.annotate(f'Crossover ≈ {cross_x:.0f}%',
                         xy=(cross_x, cross_y),
                         xytext=(cross_x + 5, cross_y + 3),
                         fontsize=9, style='italic',
                         arrowprops=dict(arrowstyle='->', color='gray'))
            break

    fig.tight_layout()
    save_fig(fig, 'fig10_crossover_analysis')


def fig_professional_sequences():
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
    bars1 = ax.bar(x - width/2, trad, width, label=TRAD_LABEL, **TRAD_BAR)
    bars2 = ax.bar(x + width/2, world, width, label=WORLD_LABEL, **WORLD_BAR)

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
    save_fig(fig, 'fig11_professional_sequences')


def fig_model_robustness():
    """
    Combined robustness figure: score spread under independent vs momentum
    models, and the SD gap across all three simulation models (independent,
    momentum, Markov chain with spare difficulty). Shows that more realistic
    models amplify the traditional scoring advantage.
    """
    tier_names = list(SKILL_TIERS.keys())
    n_games = 30_000
    momentum = 0.05

    models = {
        'Independent': lambda p: simulate_games(
            n_games, p['p_strike'], p['p_spare'], p['pin_mean']),
        'Momentum': lambda p: simulate_games_momentum(
            n_games, p['p_strike'], p['p_spare'], p['pin_mean']),
        'Markov + spare\ndifficulty': lambda p: simulate_games_markov(
            n_games, p['p_strike'], p['p_spare'], p['pin_mean']),
    }

    # Single pass: store trad/world SDs per model per tier
    sds = {m: {'trad': [], 'world': []} for m in models}
    for name, params in SKILL_TIERS.items():
        for mname, mfn in models.items():
            t, w = mfn(params)
            sds[mname]['trad'].append(np.std(t))
            sds[mname]['world'].append(np.std(w))

    x = np.arange(len(tier_names))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: score SD by tier — independent vs momentum
    ax1.plot(tier_names, sds['Independent']['trad'], '-', marker='o',
             color=TRAD_COLOR, label='Trad. (independent)', markersize=7)
    ax1.plot(tier_names, sds['Momentum']['trad'], marker='^', color=TRAD_COLOR,
             label='Trad. (momentum)', markersize=7, alpha=0.7, linestyle=':')
    ax1.plot(tier_names, sds['Independent']['world'], marker='s',
             color=WORLD_COLOR, label='WB (independent)', markersize=7,
             linestyle='--')
    ax1.plot(tier_names, sds['Momentum']['world'], marker='v',
             color=WORLD_COLOR, label='WB (momentum)', markersize=7,
             alpha=0.7, linestyle=':')
    ax1.set_ylabel('Score Standard Deviation')
    ax1.set_title('Score Spread: Independent vs Momentum Model')
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=15)

    # Panel 2: SD gap (trad - world) — independent vs momentum
    gap_indep = [t - w for t, w in zip(sds['Independent']['trad'],
                                       sds['Independent']['world'])]
    gap_mom = [t - w for t, w in zip(sds['Momentum']['trad'],
                                     sds['Momentum']['world'])]

    ax2.bar(x - 0.18, gap_indep, 0.35, label='Independent model', **TRAD_BAR)
    ax2.bar(x + 0.18, gap_mom, 0.35, label='Momentum model',
            color='#555555', edgecolor='#333333', hatch='...', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_names, fontsize=9, rotation=15)
    ax2.set_ylabel('SD Gap (Traditional − World Bowling)')
    ax2.set_title('Spread Advantage: Independent vs Momentum')
    ax2.legend(fontsize=8)
    ax2.axhline(0, color='gray', linewidth=0.5)

    # Panel 3: SD gap across all three models
    gaps = {m: [t - w for t, w in zip(sds[m]['trad'], sds[m]['world'])]
            for m in models}
    for mname in models:
        if mname.startswith('Markov'):
            for name, g in zip(tier_names, gaps[mname]):
                print(f'  {name:15s}: Markov gap={g:+.1f}')

    width = 0.25
    colors = ['#1a1a1a', '#666666', '#aaaaaa']
    hatches = ['', '///', '...']
    for i, (mname, gap_vals) in enumerate(gaps.items()):
        ax3.bar(x + (i - 1) * width, gap_vals, width,
                label=mname, color=colors[i], edgecolor='#333',
                hatch=hatches[i], alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(tier_names, fontsize=9, rotation=15)
    ax3.set_ylabel('SD Gap (Traditional − World Bowling)')
    ax3.set_title('Spread Advantage: Three Models')
    ax3.legend(fontsize=8)
    ax3.axhline(0, color='gray', linewidth=0.5)

    fig.tight_layout()
    save_fig(fig, 'fig12_model_robustness')


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print('Generating simulation figures...')
    fig_tier_distributions()
    fig_mean_scores_by_tier()
    fig_crossover()
    fig_professional_sequences()
    fig_model_robustness()
    print('Done — all simulation figures saved.')


if __name__ == '__main__':
    main()
