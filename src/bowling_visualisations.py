#!/usr/bin/env python3
"""
bowling_visualisations.py

Generate publication-quality figures for the bowling scoring paper.

Figures:
  1. Score distribution overlay (traditional vs World Bowling)
  2. Score distribution overlay — high-score tail (200+)
  3. Reward gradient for consecutive strikes
  4. Cumulative distribution comparison
  5. Log-scale distribution comparison for high scores
"""

import os
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────

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

TRAD_COLOR = '#2166ac'
WORLD_COLOR = '#b2182b'
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_distributions():
    """Load score distributions from CSV."""
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
    """Convert raw counts to probabilities."""
    return counts / counts.sum()


# ── Figure 1: Full Distribution Overlay ──────────────────────────────────────

def fig_distribution_overlay(scores, trad_p, world_p):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(scores, trad_p * 100, color=TRAD_COLOR, linewidth=1.2,
            label='Traditional', alpha=0.85)
    ax.plot(scores, world_p * 100, color=WORLD_COLOR, linewidth=1.2,
            label='World Bowling', alpha=0.85)

    ax.set_xlabel('Score')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Score Distributions: Traditional vs World Bowling')
    ax.set_xlim(0, 300)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Mark modes
    trad_mode = scores[np.argmax(trad_p)]
    world_mode = scores[np.argmax(world_p)]
    ax.axvline(trad_mode, color=TRAD_COLOR, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axvline(world_mode, color=WORLD_COLOR, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.annotate(f'Mode={trad_mode}', xy=(trad_mode, np.max(trad_p)*100),
                xytext=(trad_mode + 8, np.max(trad_p)*100),
                fontsize=8, color=TRAD_COLOR)
    ax.annotate(f'Mode={world_mode}', xy=(world_mode, np.max(world_p)*100),
                xytext=(world_mode - 35, np.max(world_p)*100 * 0.95),
                fontsize=8, color=WORLD_COLOR)

    path = os.path.join(FIGURES_DIR, 'fig1_distribution_overlay.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved: {path}')
    # Also save PNG for quick viewing
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(scores, trad_p * 100, color=TRAD_COLOR, linewidth=1.2,
             label='Traditional', alpha=0.85)
    ax2.plot(scores, world_p * 100, color=WORLD_COLOR, linewidth=1.2,
             label='World Bowling', alpha=0.85)
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Probability (%)')
    ax2.set_title('Score Distributions: Traditional vs World Bowling')
    ax2.set_xlim(0, 300)
    ax2.legend()
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.axvline(trad_mode, color=TRAD_COLOR, linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.axvline(world_mode, color=WORLD_COLOR, linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.annotate(f'Mode={trad_mode}', xy=(trad_mode, np.max(trad_p)*100),
                 xytext=(trad_mode + 8, np.max(trad_p)*100),
                 fontsize=8, color=TRAD_COLOR)
    ax2.annotate(f'Mode={world_mode}', xy=(world_mode, np.max(world_p)*100),
                 xytext=(world_mode - 35, np.max(world_p)*100 * 0.95),
                 fontsize=8, color=WORLD_COLOR)
    png_path = os.path.join(FIGURES_DIR, 'fig1_distribution_overlay.png')
    fig2.savefig(png_path)
    plt.close(fig2)
    print(f'  Saved: {png_path}')


# ── Figure 2: High-Score Tail ────────────────────────────────────────────────

def fig_high_score_tail(scores, trad_p, world_p):
    mask = scores >= 150
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(scores[mask], trad_p[mask], color=TRAD_COLOR, linewidth=1.2,
                label='Traditional', alpha=0.85)
    ax.semilogy(scores[mask], world_p[mask], color=WORLD_COLOR, linewidth=1.2,
                label='World Bowling', alpha=0.85)

    # Highlight the impossible gap
    gap_scores = np.arange(290, 300)
    for s in gap_scores:
        idx = np.where(scores == s)[0][0]
        if world_p[idx] == 0:
            ax.axvline(s, color=WORLD_COLOR, alpha=0.15, linewidth=4)

    ax.annotate('Scores 290–299\nimpossible under\nWorld Bowling',
                xy=(294, 1e-20), fontsize=8, color=WORLD_COLOR,
                ha='center', style='italic')

    ax.set_xlabel('Score')
    ax.set_ylabel('Probability (log scale)')
    ax.set_title('High-Score Tail: Traditional vs World Bowling')
    ax.set_xlim(150, 302)
    ax.legend()

    path = os.path.join(FIGURES_DIR, 'fig2_high_score_tail.pdf')
    fig.savefig(path)
    png_path = os.path.join(FIGURES_DIR, 'fig2_high_score_tail.png')
    fig.savefig(png_path)
    plt.close(fig)
    print(f'  Saved: {path}')
    print(f'  Saved: {png_path}')


# ── Figure 3: Reward Gradient ────────────────────────────────────────────────

def fig_reward_gradient():
    """Marginal value of Nth consecutive strike under each system."""
    from bowling_distributions import traditional_distribution, world_bowling_distribution
    from bowling_analysis import score_traditional, score_world

    def game_with_n_strikes(n):
        balls = [10] * n
        remaining = 10 - n
        for _ in range(remaining):
            balls += [5, 4]
        if n == 10:
            balls = [10] * 12
        elif n == 9:
            balls[-2:] = [10, 5, 4]
        return balls

    strikes = list(range(11))
    trad_scores = []
    world_scores = []
    for n in strikes:
        balls = game_with_n_strikes(n)
        trad_scores.append(score_traditional(balls))
        world_scores.append(score_world(balls))

    trad_marginal = [0] + [trad_scores[i] - trad_scores[i-1] for i in range(1, 11)]
    world_marginal = [0] + [world_scores[i] - world_scores[i-1] for i in range(1, 11)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: total scores
    ax1.plot(strikes, trad_scores, 'o-', color=TRAD_COLOR, label='Traditional',
             markersize=6)
    ax1.plot(strikes, world_scores, 's-', color=WORLD_COLOR, label='World Bowling',
             markersize=6)
    ax1.set_xlabel('Consecutive Strikes (frames 1–N)')
    ax1.set_ylabel('Total Game Score')
    ax1.set_title('Total Score vs Consecutive Strikes')
    ax1.legend()
    ax1.set_xticks(strikes)

    # Right panel: marginal value
    x = list(range(1, 11))
    ax2.bar([i - 0.18 for i in x], trad_marginal[1:], width=0.35,
            color=TRAD_COLOR, label='Traditional', alpha=0.85)
    ax2.bar([i + 0.18 for i in x], world_marginal[1:], width=0.35,
            color=WORLD_COLOR, label='World Bowling', alpha=0.85)
    ax2.set_xlabel('Strike Number in Consecutive Run')
    ax2.set_ylabel('Marginal Score Increase')
    ax2.set_title('Marginal Value of Each Additional Strike')
    ax2.legend()
    ax2.set_xticks(x)
    ax2.axhline(21, color=WORLD_COLOR, linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.annotate('World Bowling: flat +21', xy=(7, 22), fontsize=8,
                 color=WORLD_COLOR, style='italic')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_reward_gradient.pdf')
    fig.savefig(path)
    png_path = os.path.join(FIGURES_DIR, 'fig3_reward_gradient.png')
    fig.savefig(png_path)
    plt.close(fig)
    print(f'  Saved: {path}')
    print(f'  Saved: {png_path}')


# ── Figure 4: Cumulative Distribution ────────────────────────────────────────

def fig_cumulative(scores, trad_p, world_p):
    trad_cdf = np.cumsum(trad_p)
    world_cdf = np.cumsum(world_p)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scores, trad_cdf, color=TRAD_COLOR, linewidth=1.5, label='Traditional')
    ax.plot(scores, world_cdf, color=WORLD_COLOR, linewidth=1.5, label='World Bowling')
    ax.set_xlabel('Score')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Functions')
    ax.legend()
    ax.set_xlim(0, 300)

    # Mark medians
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

    path = os.path.join(FIGURES_DIR, 'fig4_cumulative_distribution.pdf')
    fig.savefig(path)
    png_path = os.path.join(FIGURES_DIR, 'fig4_cumulative_distribution.png')
    fig.savefig(png_path)
    plt.close(fig)
    print(f'  Saved: {path}')
    print(f'  Saved: {png_path}')


# ── Figure 5: Sequence Sensitivity Visualisation ─────────────────────────────

def fig_sequence_sensitivity():
    """
    Bar chart showing how the same frame composition yields different
    traditional scores but identical World Bowling scores.
    """
    from bowling_analysis import score_traditional, score_world

    # 9 frames of strikes and spares + neutral frame 10
    strike = [10]
    spare = [5, 5]
    neutral_10 = [5, 4]

    # All sequences use the SAME frame composition: 5 strikes + 4 spares
    # Only the ORDER differs — World Bowling scores must be identical
    sequences = {
        '5X then\n4 spares': strike*5 + spare*4 + neutral_10,
        'Alt X/spare\n×4, then X': (strike + spare)*4 + strike + neutral_10,
        '4 spares\nthen 5X': spare*4 + strike*5 + neutral_10,
        'X, sp, sp, X,\nsp, X, X, sp, X': strike + spare + spare + strike + spare + strike + strike + spare + strike + neutral_10,
    }

    names = list(sequences.keys())
    trad_scores = [score_traditional(b) for b in sequences.values()]
    world_scores = [score_world(b) for b in sequences.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, trad_scores, width, color=TRAD_COLOR,
                   label='Traditional', alpha=0.85)
    bars2 = ax.bar(x + width/2, world_scores, width, color=WORLD_COLOR,
                   label='World Bowling', alpha=0.85)

    ax.set_ylabel('Game Score')
    ax.set_title('Sequence Sensitivity: Same Frames, Different Order')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom',
                fontsize=8, color=TRAD_COLOR)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom',
                fontsize=8, color=WORLD_COLOR)

    ax.set_ylim(0, max(max(trad_scores), max(world_scores)) + 25)

    path = os.path.join(FIGURES_DIR, 'fig5_sequence_sensitivity.pdf')
    fig.savefig(path)
    png_path = os.path.join(FIGURES_DIR, 'fig5_sequence_sensitivity.png')
    fig.savefig(png_path)
    plt.close(fig)
    print(f'  Saved: {path}')
    print(f'  Saved: {png_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('Loading distributions...')
    scores, trad_raw, world_raw = load_distributions()
    trad_p = to_probability(trad_raw)
    world_p = to_probability(world_raw)

    print('Generating figures...')
    fig_distribution_overlay(scores, trad_p, world_p)
    fig_high_score_tail(scores, trad_p, world_p)
    fig_reward_gradient()
    fig_cumulative(scores, trad_p, world_p)
    fig_sequence_sensitivity()
    print('Done — all figures saved to figures/')


if __name__ == '__main__':
    main()
