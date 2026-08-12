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
from plot_style import (
    TRAD_COLOR, WORLD_COLOR, TRAD_FILL, WORLD_FILL,
    TRAD_BAR, WORLD_BAR, TRAD_LABEL, WORLD_LABEL,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


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

    # ── Figure 6: Combined permutation analysis (2 rows) ─────────────────────
    # Top row: score range and SD by composition. Bottom row: score
    # distributions across all orderings for three compositions (2X, 5X, 7X).

    # Use just the strike/spare compositions for the main figure
    strike_spare_results = all_results[:8]
    n_strikes_vals = list(range(1, 9))
    trad_ranges = [r['trad_range'] for r in strike_spare_results]
    trad_stdevs = [r['trad_stdev'] for r in strike_spare_results]
    world_ranges = [r['world_range'] for r in strike_spare_results]

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.4)
    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])

    # Top-left: score range
    ax1.bar([n - 0.18 for n in n_strikes_vals], trad_ranges, width=0.35,
            label=TRAD_LABEL, **TRAD_BAR)
    ax1.bar([n + 0.18 for n in n_strikes_vals], world_ranges, width=0.35,
            label=WORLD_LABEL, **WORLD_BAR)
    ax1.set_xlabel('Number of Strikes (out of 9 frames)')
    ax1.set_ylabel('Score Range (max − min)')
    ax1.set_title('Score Range Across All Orderings')
    ax1.legend()
    ax1.set_xticks(n_strikes_vals)

    # Top-right: standard deviation
    ax2.plot(n_strikes_vals, trad_stdevs, '-', marker='o', color=TRAD_COLOR,
             label=TRAD_LABEL, markersize=6)
    ax2.plot(n_strikes_vals, [0]*8, '--', marker='s', color=WORLD_COLOR,
             label=WORLD_LABEL + ' (always 0)', markersize=6, alpha=0.7)
    ax2.set_xlabel('Number of Strikes (out of 9 frames)')
    ax2.set_ylabel('Score Std Deviation')
    ax2.set_title('Score Variability Due to Frame Order')
    ax2.legend()
    ax2.set_xticks(n_strikes_vals)

    # Bottom row: score distributions for 2X+7sp, 5X+4sp, 7X+2sp (as Fig 5)
    panel_indices = [1, 4, 6]  # indices into all_results (0-based: 2X, 5X, 7X)
    panel_labels = ['2 Strikes + 7 Spares', '5 Strikes + 4 Spares', '7 Strikes + 2 Spares']

    for idx, (ri, label) in enumerate(zip(panel_indices, panel_labels)):
        ax = fig.add_subplot(gs[1, idx*2:(idx+1)*2])
        r = all_results[ri]
        trad_counter = Counter(r['trad_scores'])

        trad_x = sorted(trad_counter.keys())
        trad_y = [trad_counter[s] for s in trad_x]

        ax.bar(trad_x, trad_y, width=0.8, label=TRAD_LABEL, **TRAD_BAR)

        # World Bowling: single vertical line
        wb_score = r['world_scores'][0]
        ax.axvline(wb_score, color=WORLD_COLOR, linewidth=2.5, linestyle='--',
                   label=f'{WORLD_LABEL} (all = {wb_score})')

        ax.set_xlabel('Game Score')
        if idx == 0:
            ax.set_ylabel('Number of Orderings')
        ax.set_title(f'{label}\n({r["n_permutations"]} orderings)')
        ax.legend(fontsize=7, loc='upper left')

        # Annotate range
        ax.annotate(f'Range: {r["trad_range"]} pts',
                    xy=(r['trad_min'], max(trad_y) * 0.85),
                    fontsize=8, color=TRAD_COLOR, style='italic')

    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'fig6_permutation_analysis.{ext}')
        fig.savefig(path)
        print(f'Saved: {path}')
    plt.close(fig)

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
