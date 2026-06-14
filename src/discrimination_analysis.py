#!/usr/bin/env python3
"""
discrimination_analysis.py

Signal-versus-noise analysis of the two scoring systems.

The score-spread (standard deviation) results in bowling_simulation.py answer
"how wide is the score distribution?" but not "how much of that width is signal
(genuine between-player skill differences) versus noise (game-to-game variation
for a fixed player)?" A scoring system can inflate the standard deviation simply
by amplifying the consequences of a single lucky or unlucky ball, which would be
noise, not discrimination.

This module decomposes score variance into between-player and within-player
components and reports the single-game reliability (an intraclass correlation,
ICC) of each system:

    ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2)

ICC is the fraction of a single game's score variance attributable to true
skill differences. Higher ICC means a single game is more informative about
which of two players is the better one. Crucially, both systems are scored on
the *same* simulated ball sequences, so any difference in ICC is due to the
scoring function alone.

Three analyses:
  1. reliability_vs_skill   — ICC as a function of skill (strike rate), where
                              "skill" varies on the primary axis (strike rate).
  2. streakiness_discrimination — players share an identical marginal strike rate
                              but differ in lag-1 autocorrelation (streakiness).
                              Tests whether each system can detect sequencing
                              ability at fixed raw skill.
  3. bootstrap_crossover    — confidence interval for the score-spread crossover
                              point via resampling.
"""

import os
import csv

import numpy as np

from bowling_simulation import (
    SKILL_TIERS, simulate_game, simulate_game_markov, simulate_game_autocorr,
)
from bowling_analysis import score_traditional, score_world

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ── Variance decomposition ────────────────────────────────────────────────────

def icc_oneway(scores):
    """
    One-way random-effects variance decomposition.

    scores: array of shape (n_players, n_games), one row per player.
    Returns dict with between-player variance, within-player variance, and the
    single-game reliability ICC(1). Uses the standard unbiased ANOVA estimator;
    the between-player component is floored at zero (a negative estimate means no
    detectable between-player signal).
    """
    scores = np.asarray(scores, dtype=float)
    n_players, n_games = scores.shape
    player_means = scores.mean(axis=1)
    grand_mean = scores.mean()

    ms_between = n_games * np.sum((player_means - grand_mean) ** 2) / (n_players - 1)
    ms_within = np.sum((scores - player_means[:, None]) ** 2) / (n_players * (n_games - 1))

    sigma_b = max((ms_between - ms_within) / n_games, 0.0)
    sigma_w = ms_within
    icc = sigma_b / (sigma_b + sigma_w) if (sigma_b + sigma_w) > 0 else 0.0
    return {'sigma_between': sigma_b, 'sigma_within': sigma_w, 'icc': icc}


def cohens_d(a, b):
    """Cohen's d between two samples (a is the 'stronger' group)."""
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


# ── Skill parameter mapping ───────────────────────────────────────────────────

def skill_params(p_strike):
    """Map a strike rate to spare rate and pin mean, matching crossover_analysis."""
    p_spare = min(0.3 + p_strike * 0.8, 0.95)
    pin_mean = min(5.0 + p_strike * 4.0, 9.0)
    return p_spare, pin_mean


def _simulate_player(rng, p_strike, p_spare, pin_mean, n_games, model='independent',
                     phi=0.0):
    """
    Simulate n_games for one player; score each game under both systems.
    Returns (trad_scores, world_scores) as equal-length arrays from the SAME
    ball sequences.
    """
    trad = np.empty(n_games)
    world = np.empty(n_games)
    k = 0
    for _ in range(n_games):
        if model == 'autocorr':
            balls = simulate_game_autocorr(rng, p_strike, p_spare, pin_mean, phi)
        elif model == 'markov':
            balls = simulate_game_markov(rng, p_strike, p_spare, pin_mean)
        else:
            balls = simulate_game(rng, p_strike, p_spare, pin_mean)
        t = score_traditional(balls)
        w = score_world(balls)
        if t is not None and w is not None:
            trad[k] = t
            world[k] = w
            k += 1
    return trad[:k], world[:k]


# ── Analysis 1: reliability (ICC) as a function of skill ──────────────────────

def reliability_vs_skill(strike_rates=None, n_players=120, n_games=60,
                         sigma_skill=0.03, seed=20240601):
    """
    At each target strike rate, build a population of players whose true strike
    rates differ by a realistic amount (sigma_skill, in strike-rate points), then
    measure how reliably one game separates them under each scoring system.

    sigma_skill = 0.03 reflects the within-neighbourhood spread of genuine ability
    among players nominally at the same level (cf. the ~53-66% posterior strike
    range across 53 PBA bowlers in VanDerwerken & Kenter 2018).
    """
    if strike_rates is None:
        strike_rates = np.linspace(0.10, 0.78, 18)

    rng = np.random.default_rng(seed)
    rows = []
    print('\n' + '=' * 78)
    print('  RELIABILITY (ICC) vs SKILL  —  signal / (signal + noise) per game')
    print('=' * 78)
    print(f'\n{"Strike%":>8} {"Trad ICC":>9} {"WB ICC":>8} '
          f'{"Trad sB":>8} {"WB sB":>8} {"Trad sW":>8} {"WB sW":>8}')
    print('-' * 70)

    for s0 in strike_rates:
        trad_rows, world_rows = [], []
        # Draw true strike rates for the player population around s0.
        true_rates = np.clip(rng.normal(s0, sigma_skill, n_players), 0.02, 0.95)
        for ps in true_rates:
            p_spare, pin_mean = skill_params(ps)
            t, w = _simulate_player(rng, ps, p_spare, pin_mean, n_games,
                                    model='independent')
            m = min(len(t), len(w), n_games)
            trad_rows.append(t[:m])
            world_rows.append(w[:m])

        # Trim to common length so the matrix is rectangular.
        m = min(len(r) for r in trad_rows)
        trad_mat = np.array([r[:m] for r in trad_rows])
        world_mat = np.array([r[:m] for r in world_rows])

        td = icc_oneway(trad_mat)
        wd = icc_oneway(world_mat)
        rows.append({
            'strike_rate': float(s0),
            'trad_icc': td['icc'], 'world_icc': wd['icc'],
            'trad_sigma_b': td['sigma_between'], 'world_sigma_b': wd['sigma_between'],
            'trad_sigma_w': td['sigma_within'], 'world_sigma_w': wd['sigma_within'],
        })
        print(f'{s0*100:>7.1f}% {td["icc"]:>9.3f} {wd["icc"]:>8.3f} '
              f'{td["sigma_between"]:>8.1f} {wd["sigma_between"]:>8.1f} '
              f'{td["sigma_within"]:>8.1f} {wd["sigma_within"]:>8.1f}')

    return rows


# ── Analysis 2: streakiness discrimination at fixed marginal skill ────────────

def streakiness_discrimination(p_strike=0.66, phi_max=0.35, n_players=140,
                               n_games=80, seed=20240602):
    """
    Hold the marginal strike rate fixed and vary lag-1 autocorrelation (phi)
    across a player population. Because World Bowling scoring depends only on the
    multiset of frame outcomes, two players with the same marginal strike rate
    have (in expectation) the same World Bowling mean regardless of streakiness,
    so World Bowling's between-player signal collapses toward zero. Traditional
    scoring rewards consecutive strikes, so streakier players score higher and a
    genuine between-player signal appears.

    Returns (summary dict, paired-comparison dict).
    """
    p_spare, pin_mean = skill_params(p_strike)
    rng = np.random.default_rng(seed)

    print('\n' + '=' * 78)
    print(f'  STREAKINESS DISCRIMINATION AT FIXED SKILL '
          f'(marginal strike rate = {p_strike*100:.0f}%)')
    print('=' * 78)

    # Population: players identical in marginal skill, differing only in phi.
    phis = rng.uniform(0.0, phi_max, n_players)
    trad_rows, world_rows = [], []
    for phi in phis:
        t, w = _simulate_player(rng, p_strike, p_spare, pin_mean, n_games,
                                model='autocorr', phi=phi)
        m = min(len(t), len(w), n_games)
        trad_rows.append(t[:m])
        world_rows.append(w[:m])

    m = min(len(r) for r in trad_rows)
    trad_mat = np.array([r[:m] for r in trad_rows])
    world_mat = np.array([r[:m] for r in world_rows])

    td = icc_oneway(trad_mat)
    wd = icc_oneway(world_mat)

    # Correlation of player-mean score with true streakiness phi.
    trad_means = trad_mat.mean(axis=1)
    world_means = world_mat.mean(axis=1)
    trad_corr = np.corrcoef(phis, trad_means)[0, 1]
    world_corr = np.corrcoef(phis, world_means)[0, 1]

    print(f'\nPopulation ICC (between-player signal from streakiness alone):')
    print(f'  Traditional : ICC = {td["icc"]:.3f}  '
          f'(sigma_between = {td["sigma_between"]:.1f}, sigma_within = {td["sigma_within"]:.1f})')
    print(f'  World Bowl. : ICC = {wd["icc"]:.3f}  '
          f'(sigma_between = {wd["sigma_between"]:.1f}, sigma_within = {wd["sigma_within"]:.1f})')
    print(f'\nCorrelation of player mean score with true streakiness phi:')
    print(f'  Traditional : r = {trad_corr:+.3f}')
    print(f'  World Bowl. : r = {world_corr:+.3f}')

    summary = {
        'p_strike': p_strike,
        'trad_icc': td['icc'], 'world_icc': wd['icc'],
        'trad_sigma_b': td['sigma_between'], 'world_sigma_b': wd['sigma_between'],
        'trad_sigma_w': td['sigma_within'], 'world_sigma_w': wd['sigma_within'],
        'trad_corr_phi': trad_corr, 'world_corr_phi': world_corr,
    }

    # Paired head-to-head: a low-streak vs a high-streak player, same marginal.
    n_pair = 6000
    t_lo, w_lo = _simulate_player(rng, p_strike, p_spare, pin_mean, n_pair,
                                  model='autocorr', phi=0.0)
    t_hi, w_hi = _simulate_player(rng, p_strike, p_spare, pin_mean, n_pair,
                                  model='autocorr', phi=phi_max)
    pair = {
        'phi_low': 0.0, 'phi_high': phi_max,
        'trad_mean_low': float(t_lo.mean()), 'trad_mean_high': float(t_hi.mean()),
        'world_mean_low': float(w_lo.mean()), 'world_mean_high': float(w_hi.mean()),
        'trad_d': cohens_d(t_hi, t_lo), 'world_d': cohens_d(w_hi, w_lo),
        # arrays kept for plotting
        '_t_lo': t_lo, '_t_hi': t_hi, '_w_lo': w_lo, '_w_hi': w_hi,
    }
    print(f'\nHead-to-head: low streak (phi=0) vs high streak (phi={phi_max}), '
          f'same {p_strike*100:.0f}% strike rate:')
    print(f'  Traditional : {pair["trad_mean_low"]:.1f} -> {pair["trad_mean_high"]:.1f} '
          f'(Cohen d = {pair["trad_d"]:+.3f})')
    print(f'  World Bowl. : {pair["world_mean_low"]:.1f} -> {pair["world_mean_high"]:.1f} '
          f'(Cohen d = {pair["world_d"]:+.3f})')

    details = {
        'phis': phis,
        'trad_player_means': trad_means,
        'world_player_means': world_means,
    }
    return summary, pair, details


# ── Analysis 3: bootstrap CI for the score-spread crossover point ─────────────

def _interp_crossing(x, diff):
    """First x where `diff` (trad SD - world SD) crosses from <=0 to >0."""
    for i in range(1, len(diff)):
        if diff[i - 1] <= 0 < diff[i]:
            x0, x1 = x[i - 1], x[i]
            d0, d1 = diff[i - 1], diff[i]
            return x0 + (x1 - x0) * (-d0) / (d1 - d0)
    return None


def bootstrap_crossover(strike_rates=None, n_games=20_000, n_boot=500,
                        seed=20240603):
    """
    Estimate the crossover strike rate (where traditional score SD overtakes
    World Bowling SD) with a bootstrap confidence interval. At each grid point we
    simulate once, then resample games to recompute the SD-difference curve and
    locate its zero crossing.
    """
    if strike_rates is None:
        strike_rates = np.linspace(0.30, 0.75, 19)

    print('\n' + '=' * 78)
    print('  BOOTSTRAP CONFIDENCE INTERVAL FOR THE CROSSOVER POINT')
    print('=' * 78)

    trad_by_rate, world_by_rate = [], []
    for i, ps in enumerate(strike_rates):
        p_spare, pin_mean = skill_params(ps)
        rng = np.random.default_rng(seed + i)
        t, w = _simulate_player(rng, ps, p_spare, pin_mean, n_games,
                                model='independent')
        m = min(len(t), len(w))
        trad_by_rate.append(t[:m])
        world_by_rate.append(w[:m])

    # Point estimate from full samples.
    point_diff = np.array([np.std(trad_by_rate[i]) - np.std(world_by_rate[i])
                           for i in range(len(strike_rates))])
    point = _interp_crossing(strike_rates, point_diff)

    boot_rng = np.random.default_rng(seed + 999)
    crossings = []
    for _ in range(n_boot):
        diff = np.empty(len(strike_rates))
        for i in range(len(strike_rates)):
            n = len(trad_by_rate[i])
            idx = boot_rng.integers(0, n, n)
            diff[i] = np.std(trad_by_rate[i][idx]) - np.std(world_by_rate[i][idx])
        c = _interp_crossing(strike_rates, diff)
        if c is not None:
            crossings.append(c)

    crossings = np.array(crossings)
    lo, hi = np.percentile(crossings, [2.5, 97.5])
    result = {
        'point': float(point) if point is not None else None,
        'ci_low': float(lo), 'ci_high': float(hi),
        'median': float(np.median(crossings)),
        'n_valid': len(crossings), 'n_boot': n_boot,
    }
    print(f'\nCrossover strike rate: {result["point"]*100:.1f}%  '
          f'(95% bootstrap CI: {lo*100:.1f}% - {hi*100:.1f}%, '
          f'{len(crossings)}/{n_boot} replicates with a crossing)')
    return result


# ── Face validity: reproduce published aggregate moments ──────────────────────

def validate_against_published(mu=0.54, sigma_skill=0.08, n_bowlers=400,
                              n_games=200, seed=20240604):
    """
    Face-validity check against real data. @mccarthy2011 reports a mean game
    score of 205.65 (SD 31.91) across 278,579 PBA National Tour games, pooled
    over bowlers of differing ability. We simulate an analogous mixture: a
    population of bowlers whose strike rates are drawn from N(mu, sigma_skill),
    each playing many games, scored under traditional rules. If the pooled mean
    and SD land near the published values, the parametric model reproduces the
    real aggregate distribution, not just its mean.
    """
    rng = np.random.default_rng(seed)
    rates = np.clip(rng.normal(mu, sigma_skill, n_bowlers), 0.35, 0.82)
    scores = []
    for r in rates:
        p_spare, pin_mean = skill_params(r)
        t, _ = _simulate_player(rng, r, p_spare, pin_mean, n_games,
                                model='independent')
        scores.append(t)
    scores = np.concatenate(scores)

    result = {
        'sim_mean': float(scores.mean()), 'sim_sd': float(scores.std()),
        'published_mean': 205.65, 'published_sd': 31.91,
        'mix_mu': mu, 'mix_sigma_skill': sigma_skill,
    }
    print('\n' + '=' * 78)
    print('  FACE VALIDITY: SIMULATED MIXTURE vs PUBLISHED PBA MOMENTS')
    print('=' * 78)
    print(f'\nMixture (mean strike rate {mu:.0%}, ability SD {sigma_skill*100:.0f} pts), '
          f'{n_bowlers} bowlers x {n_games} games:')
    print(f'  Simulated traditional : mean = {result["sim_mean"]:.1f}, '
          f'SD = {result["sim_sd"]:.1f}')
    print(f'  McCarthy (2011) PBA   : mean = 205.65, SD = 31.91')
    return result


# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(reliability_rows, streak_summary, crossover_ci):
    os.makedirs(DATA_DIR, exist_ok=True)

    path = os.path.join(DATA_DIR, 'reliability_vs_skill.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(reliability_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reliability_rows)
    print(f'\nSaved: {path}')

    path = os.path.join(DATA_DIR, 'streakiness_discrimination.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(streak_summary.keys()))
        writer.writeheader()
        writer.writerow(streak_summary)
    print(f'Saved: {path}')

    path = os.path.join(DATA_DIR, 'crossover_ci.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(crossover_ci.keys()))
        writer.writeheader()
        writer.writerow(crossover_ci)
    print(f'Saved: {path}')


def save_validation(validation):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, 'validation_moments.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(validation.keys()))
        writer.writeheader()
        writer.writerow(validation)
    print(f'Saved: {path}')


def main():
    reliability_rows = reliability_vs_skill()
    streak_summary, _pair, _details = streakiness_discrimination()
    crossover_ci = bootstrap_crossover()
    validation = validate_against_published()
    save_results(reliability_rows, streak_summary, crossover_ci)
    save_validation(validation)
    return reliability_rows, streak_summary, crossover_ci, validation


if __name__ == '__main__':
    main()
