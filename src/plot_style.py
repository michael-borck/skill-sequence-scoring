"""
plot_style.py

Shared plotting style for all figures in the bowling scoring paper.

Palette is designed to be distinguishable in both colour and greyscale
(B&W printing). Traditional = dark (black/charcoal), World Bowling = medium
grey with distinct line styles, markers, and bar hatching.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Colours (high lightness contrast for B&W) ────────────────────────────────
# Traditional: dark charcoal — prints as near-black
# World Bowling: medium steel blue — prints as distinct mid-grey
TRAD_COLOR = '#1a1a1a'
WORLD_COLOR = '#888888'

# For fills/bars where we need more separation
TRAD_FILL = '#2c2c2c'
WORLD_FILL = '#aaaaaa'

# ── Line & marker styles ─────────────────────────────────────────────────────
TRAD_LINE = {'color': TRAD_COLOR, 'linestyle': '-', 'linewidth': 1.5}
WORLD_LINE = {'color': WORLD_COLOR, 'linestyle': '--', 'linewidth': 1.5}

TRAD_MARKER = {'color': TRAD_COLOR, 'marker': 'o', 'markersize': 6}
WORLD_MARKER = {'color': WORLD_COLOR, 'marker': 's', 'markersize': 6}

# ── Bar hatching ─────────────────────────────────────────────────────────────
TRAD_BAR = {'color': TRAD_FILL, 'edgecolor': TRAD_COLOR, 'alpha': 0.85}
WORLD_BAR = {'color': WORLD_FILL, 'edgecolor': WORLD_COLOR, 'hatch': '///', 'alpha': 0.85}

# ── Labels ───────────────────────────────────────────────────────────────────
TRAD_LABEL = 'Traditional'
WORLD_LABEL = 'World Bowling'

# ── Global rcParams ──────────────────────────────────────────────────────────
def apply_style():
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

apply_style()
