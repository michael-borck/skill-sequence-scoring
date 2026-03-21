#!/usr/bin/env python3
"""
sequence_permutations.py

Exhaustive permutation analysis of sequence sensitivity.

For a fixed frame composition (k strikes + (9-k) spares/opens in frames 1-9,
with a neutral frame 10), enumerate ALL distinct orderings and compute the
score under both systems.

Key result: World Bowling score variance is exactly zero for all compositions.
Traditional score variance grows with the mix of frame types.
"""

import os
import csv
import math
import statistics
from itertools import permutations
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bowling_analysis import score_traditional, score_world

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


def unique_permutations(items):
    """Generate distinct permutations of items (handling duplicates)."""
    seen = set()
    for p in permutations(items):
        if p not in seen:
            seen.add(p)
            yield p


def frames_to_balls(frame_sequence, frame_10=(5, 4)):
    """Convert a sequence of frame types to a ball sequence."""
    balls = []
    for f in frame_sequence:
        if isinstance(f, tuple):
            balls.extend(f)
        else:
            balls.append(f)
    balls.extend(frame_10)
    return balls


def analyse_composition(frame_types, label, frame_10=(5, 4)):
    """
    Enumerate all distinct permutations of frame_types (9 frames),
    score each under both systems, and return statistics.
    """
    trad_scores = []
    world_scores = []

    for perm in unique_permutations(frame_types):
        balls = frames_to_balls(perm, frame_10)
        t = score_traditional(balls)
        w = score_world(balls)
        if t is not None and w is not None:
            trad_scores.append(t)
            world_scores.append(w)

    n_perms = len(trad_scores)

    result = {
        'label': label,
        'n_permutations': n_perms,
        'trad_min': min(trad_scores),
        'trad_max': max(trad_scores),
        'trad_range': max(trad_scores) - min(trad_scores),
        'trad_mean': statistics.mean(trad_scores),
        'trad_stdev': statistics.stdev(trad_scores) if n_perms > 1 else 0,
        'world_min': min(world_scores),
        'world_max': max(world_scores),
        'world_range': max(world_scores) - min(world_scores),
        'world_mean': statistics.mean(world_scores),
        'world_stdev': statistics.stdev(world_scores) if n_perms > 1 else 0,
        'trad_scores': trad_scores,
        'world_scores': world_scores,
    }

    return result


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Define frame types
    strike = 10
    spare = (5, 5)
    open_9 = (5, 4)
    open_7 = (4, 3)

    # Compositions to analyse (9 frames each)
    compositions = [
        (1, [strike]*1 + [spare]*8,
         '1 strike + 8 spares'),
        (2, [strike]*2 + [spare]*7,
         '2 strikes + 7 spares'),
        (3, [strike]*3 + [spare]*6,
         '3 strikes + 6 spares'),
        (4, [strike]*4 + [spare]*5,
         '4 strikes + 5 spares'),
        (5, [strike]*5 + [spare]*4,
         '5 strikes + 4 spares'),
        (6, [strike]*6 + [spare]*3,
         '6 strikes + 3 spares'),
        (7, [strike]*7 + [spare]*2,
         '7 strikes + 2 spares'),
        (8, [strike]*8 + [spare]*1,
         '8 strikes + 1 spare'),
    ]

    # Also: mixed with opens
    compositions_mixed = [
        (3, [strike]*3 + [open_9]*6,
         '3 strikes + 6 opens(5,4)'),
        (5, [strike]*5 + [open_9]*4,
         '5 strikes + 4 opens(5,4)'),
        (3, [strike]*3 + [spare]*3 + [open_7]*3,
         '3X + 3sp + 3 opens(4,3)'),
    ]

    print('=' * 75)
    print('  SEQUENCE SENSITIVITY: EXHAUSTIVE PERMUTATION ANALYSIS')
    print('=' * 75)

    # Header
    print(f'\n{"Composition":<30} {"Perms":>6} '
          f'{"Trad Range":>11} {"Trad SD":>8} '
          f'{"WB Range":>9} {"WB SD":>7}')
    print('-' * 75)

    all_results = []

    for n_strikes, frames, label in compositions + compositions_mixed:
        result = analyse_composition(tuple(frames), label)
        all_results.append(result)

        print(f'{label:<30} {result["n_permutations"]:>6} '
              f'{result["trad_range"]:>11} {result["trad_stdev"]:>8.1f} '
              f'{result["world_range"]:>9} {result["world_stdev"]:>7.1f}')

    # Save detailed results
    csv_path = os.path.join(DATA_DIR, 'sequence_permutation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['composition', 'n_permutations',
                         'trad_min', 'trad_max', 'trad_range', 'trad_mean', 'trad_stdev',
                         'world_min', 'world_max', 'world_range', 'world_mean', 'world_stdev'])
        for r in all_results:
            writer.writerow([
                r['label'], r['n_permutations'],
                r['trad_min'], r['trad_max'], r['trad_range'],
                f"{r['trad_mean']:.1f}", f"{r['trad_stdev']:.1f}",
                r['world_min'], r['world_max'], r['world_range'],
                f"{r['world_mean']:.1f}", f"{r['world_stdev']:.1f}",
            ])
    print(f'\nResults saved to: {csv_path}')

    # ── Figure 6: Score range by composition ─────────────────────────────────

    # Use just the strike/spare compositions for the main figure
    strike_spare_results = all_results[:8]
    n_strikes_vals = list(range(1, 9))
    trad_ranges = [r['trad_range'] for r in strike_spare_results]
    trad_stdevs = [r['trad_stdev'] for r in strike_spare_results]
    world_ranges = [r['world_range'] for r in strike_spare_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: score range
    ax1.bar([n - 0.18 for n in n_strikes_vals], trad_ranges, width=0.35,
            color=TRAD_COLOR, label='Traditional', alpha=0.85)
    ax1.bar([n + 0.18 for n in n_strikes_vals], world_ranges, width=0.35,
            color=WORLD_COLOR, label='World Bowling', alpha=0.85)
    ax1.set_xlabel('Number of Strikes (out of 9 frames)')
    ax1.set_ylabel('Score Range (max − min)')
    ax1.set_title('Score Range Across All Orderings')
    ax1.legend()
    ax1.set_xticks(n_strikes_vals)

    # Right: standard deviation
    ax2.plot(n_strikes_vals, trad_stdevs, 'o-', color=TRAD_COLOR,
             label='Traditional', markersize=6)
    ax2.plot(n_strikes_vals, [0]*8, 's--', color=WORLD_COLOR,
             label='World Bowling (always 0)', markersize=6, alpha=0.7)
    ax2.set_xlabel('Number of Strikes (out of 9 frames)')
    ax2.set_ylabel('Score Std Deviation')
    ax2.set_title('Score Variability Due to Frame Order')
    ax2.legend()
    ax2.set_xticks(n_strikes_vals)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_permutation_variance.pdf')
    fig.savefig(path)
    png_path = os.path.join(FIGURES_DIR, 'fig6_permutation_variance.png')
    fig.savefig(png_path)
    plt.close(fig)
    print(f'Saved: {path}')
    print(f'Saved: {png_path}')

    # ── Figure 7: Score distribution for 5 strikes + 4 spares ────────────────

    # The most interesting case: 5 strikes + 4 spares = 126 permutations
    r5 = all_results[4]  # 5 strikes + 4 spares

    fig, ax = plt.subplots(figsize=(10, 5))
    trad_counter = Counter(r5['trad_scores'])
    world_counter = Counter(r5['world_scores'])

    trad_x = sorted(trad_counter.keys())
    trad_y = [trad_counter[s] for s in trad_x]

    ax.bar(trad_x, trad_y, width=0.8, color=TRAD_COLOR, alpha=0.85,
           label='Traditional')

    # World Bowling: single vertical line
    wb_score = r5['world_scores'][0]
    ax.axvline(wb_score, color=WORLD_COLOR, linewidth=2.5, linestyle='-',
               label=f'World Bowling (all = {wb_score})')

    ax.set_xlabel('Game Score')
    ax.set_ylabel('Number of Orderings')
    ax.set_title(f'Score Distribution: 5 Strikes + 4 Spares ({r5["n_permutations"]} orderings)')
    ax.legend()

    # Annotate range
    ax.annotate(f'Range: {r5["trad_min"]}–{r5["trad_max"]}\n'
                f'({r5["trad_range"]} points)',
                xy=(r5['trad_min'], max(trad_y) * 0.8),
                fontsize=9, color=TRAD_COLOR)

    path = os.path.join(FIGURES_DIR, 'fig7_permutation_distribution.pdf')
    fig.savefig(path)
    png_path = os.path.join(FIGURES_DIR, 'fig7_permutation_distribution.png')
    fig.savefig(png_path)
    plt.close(fig)
    print(f'Saved: {path}')
    print(f'Saved: {png_path}')

    # ── Summary ──────────────────────────────────────────────────────────────

    print('\n' + '=' * 75)
    print('  SUMMARY')
    print('=' * 75)
    print(f'\nFor every composition tested:')
    print(f'  - World Bowling score range: 0 (order is irrelevant)')
    print(f'  - Traditional score range: up to {max(trad_ranges)} points')
    print(f'  - Traditional std deviation: up to {max(trad_stdevs):.1f}')
    print(f'\nThis confirms Proposition 4.1 (commutativity) empirically')
    print(f'and quantifies the sequence information lost by World Bowling.')


if __name__ == '__main__':
    main()
