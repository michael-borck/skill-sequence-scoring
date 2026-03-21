#!/usr/bin/env python3
"""
bowling_simulation.py

Monte Carlo simulation of bowling games at different skill levels.

Model:
  Each ball is modelled with skill-dependent probabilities:
    - Strike probability (p_strike): probability of knocking all 10 pins on first ball
    - Spare probability (p_spare): probability of converting remaining pins on second ball
    - Pin distribution: when not striking, first ball pins follow a truncated
      distribution shaped by skill level

  Skill tiers (calibrated from USBC/PBA statistics):
    - Recreational:  p_strike ≈ 0.15, p_spare ≈ 0.30
    - Club:          p_strike ≈ 0.30, p_spare ≈ 0.50
    - Competitive:   p_strike ≈ 0.50, p_spare ≈ 0.70
    - Elite:         p_strike ≈ 0.65, p_spare ≈ 0.85
    - Professional:  p_strike ≈ 0.75, p_spare ≈ 0.90

Each game is scored under both traditional and World Bowling systems,
enabling direct paired comparison.
"""

import os
import csv
import math
import statistics
from collections import defaultdict

import numpy as np

from bowling_analysis import score_traditional, score_world


# ── Player Model ─────────────────────────────────────────────────────────────

SKILL_TIERS = {
    'Recreational': {'p_strike': 0.04, 'p_spare': 0.27, 'pin_mean': 4.2},
    'Club':         {'p_strike': 0.20, 'p_spare': 0.47, 'pin_mean': 5.5},
    'Competitive':  {'p_strike': 0.40, 'p_spare': 0.64, 'pin_mean': 6.5},
    'Elite':        {'p_strike': 0.55, 'p_spare': 0.77, 'pin_mean': 7.2},
    'Professional': {'p_strike': 0.66, 'p_spare': 0.86, 'pin_mean': 7.8},
    'Top 10':       {'p_strike': 0.73, 'p_spare': 0.92, 'pin_mean': 8.2},
}


def simulate_first_ball(rng, p_strike, pin_mean):
    """
    Simulate first ball of a frame.
    Returns number of pins knocked down (0-10).
    """
    if rng.random() < p_strike:
        return 10

    # Non-strike: draw from a beta-binomial-like distribution
    # Use a simple approach: sample from binomial(10, pin_mean/10)
    # but exclude 10 (already handled as strike)
    p_pin = min(pin_mean / 10.0, 0.95)
    pins = rng.binomial(10, p_pin)
    while pins == 10:
        pins = rng.binomial(10, p_pin)
    return int(pins)


def simulate_second_ball(rng, first_ball, p_spare):
    """
    Simulate second ball given first ball result.
    Returns number of additional pins knocked down.
    """
    remaining = 10 - first_ball
    if remaining == 0:
        return 0

    if rng.random() < p_spare:
        return remaining  # spare

    # Not a spare: knock down some of the remaining pins
    # Use uniform-ish distribution over 0..remaining-1
    pins = rng.integers(0, remaining)  # excludes remaining (not a spare)
    return int(pins)


def simulate_game(rng, p_strike, p_spare, pin_mean):
    """
    Simulate one complete bowling game.
    Returns the ball sequence (list of ints).
    """
    balls = []

    # Frames 1-9
    for _ in range(9):
        b1 = simulate_first_ball(rng, p_strike, pin_mean)
        balls.append(b1)
        if b1 < 10:
            b2 = simulate_second_ball(rng, b1, p_spare)
            balls.append(b2)

    # Frame 10: special rules
    b1 = simulate_first_ball(rng, p_strike, pin_mean)
    balls.append(b1)

    if b1 == 10:  # strike on first ball
        b2 = simulate_first_ball(rng, p_strike, pin_mean)
        balls.append(b2)
        if b2 == 10:  # second strike
            b3 = simulate_first_ball(rng, p_strike, pin_mean)
            balls.append(b3)
        else:
            b3 = simulate_second_ball(rng, b2, p_spare)
            balls.append(b3)
    else:
        b2 = simulate_second_ball(rng, b1, p_spare)
        balls.append(b2)
        if b1 + b2 == 10:  # spare
            b3 = simulate_first_ball(rng, p_strike, pin_mean)
            balls.append(b3)

    return balls


# ── Momentum Model ───────────────────────────────────────────────────────────

def simulate_game_momentum(rng, p_strike, p_spare, pin_mean, momentum=0.05):
    """
    Simulate one game with a momentum/hot-hand effect.

    After a strike, the next frame's strike probability increases by `momentum`.
    After an open frame, it decreases by `momentum`.
    After a spare, it stays unchanged.
    The effective probability is clamped to [0.01, 0.99].
    """
    balls = []
    p_eff = p_strike

    # Frames 1-9
    for _ in range(9):
        p_clamped = max(0.01, min(0.99, p_eff))
        b1 = simulate_first_ball(rng, p_clamped, pin_mean)
        balls.append(b1)
        if b1 == 10:
            p_eff = p_strike + momentum * (1 + (p_eff - p_strike) / max(momentum, 0.01))
            p_eff = min(p_eff, p_strike + 3 * momentum)  # cap the streak bonus
        elif b1 < 10:
            b2 = simulate_second_ball(rng, b1, p_spare)
            balls.append(b2)
            if b1 + b2 == 10:  # spare — neutral
                p_eff = p_strike
            else:  # open — cold
                p_eff = p_strike - momentum

    # Frame 10
    p_clamped = max(0.01, min(0.99, p_eff))
    b1 = simulate_first_ball(rng, p_clamped, pin_mean)
    balls.append(b1)

    if b1 == 10:
        p_eff = min(p_strike + 2 * momentum, 0.99)
        b2 = simulate_first_ball(rng, max(0.01, min(0.99, p_eff)), pin_mean)
        balls.append(b2)
        if b2 == 10:
            b3 = simulate_first_ball(rng, max(0.01, min(0.99, p_eff)), pin_mean)
            balls.append(b3)
        else:
            b3 = simulate_second_ball(rng, b2, p_spare)
            balls.append(b3)
    else:
        b2 = simulate_second_ball(rng, b1, p_spare)
        balls.append(b2)
        if b1 + b2 == 10:
            b3 = simulate_first_ball(rng, max(0.01, min(0.99, p_strike)), pin_mean)
            balls.append(b3)

    return balls


def simulate_games_momentum(n_games, p_strike, p_spare, pin_mean,
                            momentum=0.05, seed=42):
    """Simulate n_games with momentum model."""
    rng = np.random.default_rng(seed)
    trad_scores = []
    world_scores = []

    for _ in range(n_games):
        balls = simulate_game_momentum(rng, p_strike, p_spare, pin_mean, momentum)
        t = score_traditional(balls)
        w = score_world(balls)
        if t is not None and w is not None:
            trad_scores.append(t)
            world_scores.append(w)

    return np.array(trad_scores), np.array(world_scores)


def simulate_games(n_games, p_strike, p_spare, pin_mean, seed=42):
    """
    Simulate n_games and score under both systems.
    Returns arrays of (traditional_scores, world_bowling_scores).
    """
    rng = np.random.default_rng(seed)
    trad_scores = []
    world_scores = []

    for _ in range(n_games):
        balls = simulate_game(rng, p_strike, p_spare, pin_mean)
        t = score_traditional(balls)
        w = score_world(balls)
        if t is not None and w is not None:
            trad_scores.append(t)
            world_scores.append(w)

    return np.array(trad_scores), np.array(world_scores)


# ── Discrimination Metrics ───────────────────────────────────────────────────

def score_separation(scores_strong, scores_weak):
    """
    Compute metrics for how well scores separate two player populations.

    Returns:
        mean_gap: difference in mean scores
        overlap: fraction of games where the weaker player scores >= stronger
        cohen_d: Cohen's d effect size
    """
    mean_gap = np.mean(scores_strong) - np.mean(scores_weak)

    # Overlap: P(weak >= strong) estimated from paired samples
    n = min(len(scores_strong), len(scores_weak))
    overlap = np.mean(scores_weak[:n] >= scores_strong[:n])

    # Cohen's d
    pooled_std = np.sqrt((np.var(scores_strong) + np.var(scores_weak)) / 2)
    cohen_d = mean_gap / pooled_std if pooled_std > 0 else 0

    return {
        'mean_gap': mean_gap,
        'overlap': overlap,
        'cohen_d': cohen_d,
    }


def upset_probability(scores_a, scores_b):
    """
    Given two arrays of single-game scores from players A (stronger) and B (weaker),
    estimate P(B beats A in a single game).
    """
    n = min(len(scores_a), len(scores_b))
    return np.mean(scores_b[:n] > scores_a[:n])


# ── Main Analysis ────────────────────────────────────────────────────────────

def run_tier_comparison(n_games=50_000):
    """
    Run simulation across all skill tiers and compute discrimination metrics.
    """
    print('=' * 80)
    print('  SKILL-WEIGHTED SIMULATION: SCORING SYSTEM COMPARISON')
    print('=' * 80)

    results = {}
    tier_names = list(SKILL_TIERS.keys())

    # Simulate each tier
    print(f'\nSimulating {n_games:,} games per tier...')
    for name, params in SKILL_TIERS.items():
        trad, world = simulate_games(
            n_games, params['p_strike'], params['p_spare'], params['pin_mean']
        )
        results[name] = {
            'trad': trad,
            'world': world,
            'params': params,
        }
        print(f'  {name:15s}: trad mean={np.mean(trad):.1f} (SD={np.std(trad):.1f}), '
              f'world mean={np.mean(world):.1f} (SD={np.std(world):.1f})')

    # Score separation between adjacent tiers
    print(f'\n{"Tier Comparison":<30} {"Trad Gap":>9} {"WB Gap":>8} '
          f'{"Trad d":>7} {"WB d":>6} '
          f'{"Trad Upset%":>12} {"WB Upset%":>10}')
    print('-' * 80)

    separation_data = []
    for i in range(len(tier_names) - 1):
        strong = tier_names[i + 1]
        weak = tier_names[i]

        trad_sep = score_separation(results[strong]['trad'], results[weak]['trad'])
        world_sep = score_separation(results[strong]['world'], results[weak]['world'])

        trad_upset = upset_probability(results[strong]['trad'], results[weak]['trad'])
        world_upset = upset_probability(results[strong]['world'], results[weak]['world'])

        label = f'{weak} vs {strong}'
        print(f'{label:<30} {trad_sep["mean_gap"]:>+9.1f} {world_sep["mean_gap"]:>+8.1f} '
              f'{trad_sep["cohen_d"]:>7.2f} {world_sep["cohen_d"]:>6.2f} '
              f'{trad_upset*100:>11.1f}% {world_upset*100:>9.1f}%')

        separation_data.append({
            'comparison': label,
            'trad_gap': trad_sep['mean_gap'],
            'world_gap': world_sep['mean_gap'],
            'trad_cohen_d': trad_sep['cohen_d'],
            'world_cohen_d': world_sep['cohen_d'],
            'trad_upset': trad_upset,
            'world_upset': world_upset,
        })

    return results, separation_data


def crossover_analysis(n_games=20_000, n_points=30):
    """
    Sweep strike rate from 10% to 80% and compute discrimination metrics
    at each level. Find the crossover point where traditional scoring
    becomes meaningfully more discriminating.
    """
    print('\n' + '=' * 80)
    print('  CROSSOVER POINT ANALYSIS')
    print('  Sweeping strike rate to find where scoring system choice matters')
    print('=' * 80)

    strike_rates = np.linspace(0.10, 0.80, n_points)
    trad_stdevs = []
    world_stdevs = []
    trad_ranges = []
    world_ranges = []
    trad_means = []
    world_means = []
    discrimination_ratios = []

    # For each strike rate, simulate two populations:
    # one at that rate, one at rate + 0.05 (slightly better player)
    delta = 0.05

    print(f'\n{"Strike%":>8} {"Trad Mean":>10} {"WB Mean":>8} '
          f'{"Trad SD":>8} {"WB SD":>7} '
          f'{"Trad d":>7} {"WB d":>6} {"Ratio":>6}')
    print('-' * 70)

    for p_strike in strike_rates:
        p_spare = 0.3 + p_strike * 0.8  # spare rate scales with skill
        p_spare = min(p_spare, 0.95)
        pin_mean = 5.0 + p_strike * 4.0
        pin_mean = min(pin_mean, 9.0)

        # Base player
        trad_base, world_base = simulate_games(
            n_games, p_strike, p_spare, pin_mean, seed=42
        )
        # Slightly better player
        p_strike2 = min(p_strike + delta, 0.95)
        p_spare2 = min(p_spare + delta * 0.5, 0.98)
        pin_mean2 = min(pin_mean + 0.3, 9.5)
        trad_better, world_better = simulate_games(
            n_games, p_strike2, p_spare2, pin_mean2, seed=123
        )

        trad_sep = score_separation(trad_better, trad_base)
        world_sep = score_separation(world_better, world_base)

        trad_stdevs.append(np.std(trad_base))
        world_stdevs.append(np.std(world_base))
        trad_ranges.append(np.ptp(trad_base))
        world_ranges.append(np.ptp(world_base))
        trad_means.append(np.mean(trad_base))
        world_means.append(np.mean(world_base))

        ratio = trad_sep['cohen_d'] / world_sep['cohen_d'] if world_sep['cohen_d'] > 0 else float('inf')
        discrimination_ratios.append(ratio)

        print(f'{p_strike*100:>7.1f}% {np.mean(trad_base):>10.1f} {np.mean(world_base):>8.1f} '
              f'{np.std(trad_base):>8.1f} {np.std(world_base):>7.1f} '
              f'{trad_sep["cohen_d"]:>7.2f} {world_sep["cohen_d"]:>6.2f} {ratio:>6.2f}')

    return {
        'strike_rates': strike_rates,
        'trad_means': np.array(trad_means),
        'world_means': np.array(world_means),
        'trad_stdevs': np.array(trad_stdevs),
        'world_stdevs': np.array(world_stdevs),
        'discrimination_ratios': np.array(discrimination_ratios),
    }


def professional_sequences():
    """
    Compare specific professional-level sequences side by side.
    """
    print('\n' + '=' * 80)
    print('  PROFESSIONAL-LEVEL SEQUENCE COMPARISON')
    print('=' * 80)

    sequences = {
        'Perfect game (12 strikes)': [10]*12,
        '10 strikes + spare(7,3) + 7': [10]*10 + [7, 3, 7],
        '10 strikes + spare(5,5) + 5': [10]*10 + [5, 5, 5],
        '8 strikes + 2 spares(7,3) + 7': [10]*8 + [7, 3] + [7, 3, 7],
        'Alt strike/spare(5,5) ×5': [10, 5, 5]*3 + [10, 5, 5, 10, 5, 5, 10, 5, 5, 5],
        '4 strikes then spare(5,5) ×6 + 5': [10]*4 + [5,5]*6 + [5],
        'Spare(5,5) ×6 then 4 strikes + X,X': [5,5]*6 + [10]*4 + [10, 10],
        'All spares (5,5) + 5': [5, 5]*10 + [5],
        'All opens (8,1)': [8, 1]*10,
    }

    print(f'\n{"Sequence":<45} {"Traditional":>12} {"World":>8} {"Diff":>8}')
    print('-' * 75)
    for name, balls in sequences.items():
        t = score_traditional(balls)
        w = score_world(balls)
        if t is not None and w is not None:
            diff = t - w
            print(f'{name:<45} {t:>12} {w:>8} {diff:>+8}')
        else:
            print(f'{name:<45} {"INVALID":>12} {"INVALID":>8}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results, separation_data = run_tier_comparison(n_games=50_000)
    crossover_data = crossover_analysis(n_games=20_000, n_points=30)
    professional_sequences()

    # Save results
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save tier comparison
    csv_path = os.path.join(data_dir, 'simulation_tier_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'comparison', 'trad_gap', 'world_gap',
            'trad_cohen_d', 'world_cohen_d',
            'trad_upset', 'world_upset'])
        writer.writeheader()
        writer.writerows(separation_data)
    print(f'\nTier comparison saved to: {csv_path}')

    # Save crossover data
    csv_path2 = os.path.join(data_dir, 'simulation_crossover.csv')
    with open(csv_path2, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['strike_rate', 'trad_mean', 'world_mean',
                         'trad_stdev', 'world_stdev', 'discrimination_ratio'])
        for i in range(len(crossover_data['strike_rates'])):
            writer.writerow([
                f"{crossover_data['strike_rates'][i]:.4f}",
                f"{crossover_data['trad_means'][i]:.1f}",
                f"{crossover_data['world_means'][i]:.1f}",
                f"{crossover_data['trad_stdevs'][i]:.1f}",
                f"{crossover_data['world_stdevs'][i]:.1f}",
                f"{crossover_data['discrimination_ratios'][i]:.3f}",
            ])
    print(f'Crossover data saved to: {csv_path2}')

    return results, crossover_data


if __name__ == '__main__':
    main()
