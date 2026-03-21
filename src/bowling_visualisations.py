#!/usr/bin/env python3
"""
bowling_visualisations.py

Generate publication-quality figures for the bowling scoring paper.
Uses B&W-friendly palette from plot_style.py.

Figures:
  1. Score distribution overlay (traditional vs World Bowling)
  2. Score distribution overlay — high-score tail (200+)
  3. Reward gradient for consecutive strikes
  4. Cumulative distribution comparison
  5. Sequence sensitivity bar chart
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def load_distributions():
    scores, trad, world = [], [], []
    csv_path = os.path.join(DATA_DIR, 'bowling_distributions.csv')
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(int(row['score']))
            trad.append(int(row['traditional']))
            world.append(int(row['world_bowling']))
    return np.array(scores), np.array(trad, dtype=float), np.array(world, dtype=float)


def to_probability(counts):
    return counts / counts.sum()


# ── Figure 1: Full Distribution Overlay ──────────────────────────────────────

def fig_distribution_overlay(scores, trad_p, world_p):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(scores, trad_p * 100, label=TRAD_LABEL, **TRAD_LINE)
    ax.plot(scores, world_p * 100, label=WORLD_LABEL, **WORLD_LINE)

    ax.set_xlabel('Score')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Score Distributions: Traditional vs World Bowling')
    ax.set_xlim(0, 300)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    trad_mode = scores[np.argmax(trad_p)]
    world_mode = scores[np.argmax(world_p)]
    ax.axvline(trad_mode, color=TRAD_COLOR, linestyle=':', alpha=0.5, linewidth=0.8)
    ax.axvline(world_mode, color=WORLD_COLOR, linestyle=':', alpha=0.5, linewidth=0.8)
    ax.annotate(f'Mode={trad_mode}', xy=(trad_mode, np.max(trad_p)*100),
                xytext=(trad_mode + 8, np.max(trad_p)*100),
                fontsize=8, color=TRAD_COLOR)
    ax.annotate(f'Mode={world_mode}', xy=(world_mode, np.max(world_p)*100),
                xytext=(world_mode - 35, np.max(world_p)*100 * 0.95),
                fontsize=8, color=WORLD_COLOR)

    save_fig(fig, 'fig1_distribution_overlay')


# ── Figure 2: High-Score Tail ────────────────────────────────────────────────

def fig_high_score_tail(scores, trad_p, world_p):
    mask = scores >= 150
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(scores[mask], trad_p[mask], label=TRAD_LABEL, **TRAD_LINE)
    ax.semilogy(scores[mask], world_p[mask], label=WORLD_LABEL, **WORLD_LINE)

    # Highlight the impossible gap
    for s in range(290, 300):
        idx = np.where(scores == s)[0][0]
        if world_p[idx] == 0:
            ax.axvline(s, color=WORLD_COLOR, alpha=0.12, linewidth=4)

    ax.annotate('Scores 290–299\nimpossible under\nWorld Bowling',
                xy=(294, 1e-20), fontsize=8, color=WORLD_COLOR,
                ha='center', style='italic')

    ax.set_xlabel('Score')
    ax.set_ylabel('Probability (log scale)')
    ax.set_title('High-Score Tail: Traditional vs World Bowling')
    ax.set_xlim(150, 302)
    ax.legend()

    save_fig(fig, 'fig2_high_score_tail')


# ── Figure 3: Reward Gradient (small-multiples, 3 fill types) ────────────────

def fig_reward_gradient():
    """
    Show the reward gradient for consecutive strikes under three different
    non-strike fill patterns, demonstrating the compounding effect is
    structural and not dependent on the specific fill chosen.
    """
    from bowling_analysis import score_traditional, score_world

    fills = [
        ('Weak open (3,2)', 3, 2),
        ('Mid open (5,4)', 5, 4),
        ('Strong open (8,1)', 8, 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, (fill_label, b1, b2) in enumerate(fills):
        ax = axes[idx]

        def game_with_n_strikes(n):
            balls = [10] * n
            remaining = 10 - n
            for _ in range(remaining):
                balls += [b1, b2]
            if n == 10:
                balls = [10] * 12
            elif n == 9:
                balls[-2:] = [10, b1, b2]
            return balls

        strikes = list(range(11))
        trad_scores = [score_traditional(game_with_n_strikes(n)) for n in strikes]
        world_scores = [score_world(game_with_n_strikes(n)) for n in strikes]

        trad_marginal = [trad_scores[i] - trad_scores[i-1] for i in range(1, 11)]
        world_marginal = [world_scores[i] - world_scores[i-1] for i in range(1, 11)]

        x = list(range(1, 11))
        ax.bar([i - 0.18 for i in x], trad_marginal, width=0.35,
               label=TRAD_LABEL, **TRAD_BAR)
        ax.bar([i + 0.18 for i in x], world_marginal, width=0.35,
               label=WORLD_LABEL, **WORLD_BAR)

        wb_flat = world_marginal[0]
        ax.axhline(wb_flat, color=WORLD_COLOR, linestyle=':', alpha=0.5, linewidth=0.8)

        ax.set_xlabel('Strike Number in Run')
        ax.set_title(fill_label)
        ax.set_xticks(x)
        if idx == 0:
            ax.set_ylabel('Marginal Score Increase')
            ax.legend(fontsize=8)

    fig.suptitle('Marginal Value of Each Additional Consecutive Strike', fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig3_reward_gradient')


# ── Figure 4: Cumulative Distribution ────────────────────────────────────────

def fig_cumulative(scores, trad_p, world_p):
    trad_cdf = np.cumsum(trad_p)
    world_cdf = np.cumsum(world_p)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scores, trad_cdf, label=TRAD_LABEL, **TRAD_LINE)
    ax.plot(scores, world_cdf, label=WORLD_LABEL, **WORLD_LINE)
    ax.set_xlabel('Score')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Functions')
    ax.legend()
    ax.set_xlim(0, 300)

    trad_median_idx = np.searchsorted(trad_cdf, 0.5)
    world_median_idx = np.searchsorted(world_cdf, 0.5)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.annotate(f'Median={scores[trad_median_idx]}',
                xy=(scores[trad_median_idx], 0.5),
                xytext=(scores[trad_median_idx] + 15, 0.45),
                fontsize=8, color=TRAD_COLOR,
                arrowprops=dict(arrowstyle='->', color=TRAD_COLOR, alpha=0.6))
    ax.annotate(f'Median={scores[world_median_idx]}',
                xy=(scores[world_median_idx], 0.5),
                xytext=(scores[world_median_idx] - 40, 0.55),
                fontsize=8, color=WORLD_COLOR,
                arrowprops=dict(arrowstyle='->', color=WORLD_COLOR, alpha=0.6))

    save_fig(fig, 'fig4_cumulative_distribution')


# ── Figure 5: Sequence Sensitivity (small-multiples, 3 compositions) ─────────

def fig_sequence_sensitivity():
    """
    Show sequence sensitivity for three compositions at low/mid/high strike
    counts, demonstrating the pattern is structural — not dependent on the
    specific composition chosen.
    """
    from bowling_analysis import score_traditional, score_world

    strike = [10]
    spare = [5, 5]
    neutral_10 = [5, 4]

    compositions = [
        {
            'title': '2 Strikes + 7 Spares',
            'sequences': {
                '2X then\n7 sp': strike*2 + spare*7 + neutral_10,
                '7 sp then\n2X': spare*7 + strike*2 + neutral_10,
                'X, 3sp,\nX, 4sp': strike + spare*3 + strike + spare*4 + neutral_10,
            },
        },
        {
            'title': '5 Strikes + 4 Spares',
            'sequences': {
                '5X then\n4 sp': strike*5 + spare*4 + neutral_10,
                '4 sp then\n5X': spare*4 + strike*5 + neutral_10,
                'Alt X/sp\n×4, X': (strike + spare)*4 + strike + neutral_10,
            },
        },
        {
            'title': '7 Strikes + 2 Spares',
            'sequences': {
                '7X then\n2 sp': strike*7 + spare*2 + neutral_10,
                '2 sp then\n7X': spare*2 + strike*7 + neutral_10,
                'X,sp,\n5X,sp,X': strike + spare + strike*5 + spare + strike + neutral_10,
            },
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, comp in enumerate(compositions):
        ax = axes[idx]
        names = list(comp['sequences'].keys())
        trad_scores = [score_traditional(b) for b in comp['sequences'].values()]
        world_scores = [score_world(b) for b in comp['sequences'].values()]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, trad_scores, width, label=TRAD_LABEL, **TRAD_BAR)
        bars2 = ax.bar(x + width/2, world_scores, width, label=WORLD_LABEL, **WORLD_BAR)

        ax.set_title(comp['title'])
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        if idx == 0:
            ax.set_ylabel('Game Score')
            ax.legend(fontsize=8)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{int(bar.get_height())}', ha='center', va='bottom',
                    fontsize=7, color=TRAD_COLOR)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{int(bar.get_height())}', ha='center', va='bottom',
                    fontsize=7, color=WORLD_COLOR)

        all_scores = trad_scores + world_scores
        ax.set_ylim(0, max(all_scores) + 25)

        # Annotate the traditional range
        trad_range = max(trad_scores) - min(trad_scores)
        ax.annotate(f'Trad. range: {trad_range} pts',
                    xy=(1, min(trad_scores) - 8),
                    ha='center', fontsize=7, style='italic', color=TRAD_COLOR)

    fig.suptitle('Sequence Sensitivity: Same Frames, Different Order', fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig5_sequence_sensitivity')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('Loading distributions...')
    scores, trad_raw, world_raw = load_distributions()
    trad_p = to_probability(trad_raw)
    world_p = to_probability(world_raw)

    print('Generating figures 1–5...')
    fig_distribution_overlay(scores, trad_p, world_p)
    fig_high_score_tail(scores, trad_p, world_p)
    fig_reward_gradient()
    fig_cumulative(scores, trad_p, world_p)
    fig_sequence_sensitivity()
    print('Done — figures 1–5 saved.')


if __name__ == '__main__':
    main()
