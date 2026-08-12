#!/usr/bin/env bash
# Quarto pre-render step: copy the figures the website needs from the canonical
# figures/ dir into website/images/ so the published site (docs/) gets them.
#
# figures/ is the single source of truth (produced by the src/ plotting scripts).
# website/images/ is generated and gitignored — never edit it by hand.
set -euo pipefail

# Run from the website/ project dir (Quarto sets cwd to the project root).
cd "$(dirname "$0")"
mkdir -p images

FIGS=(
  fig8_tier_distributions
  fig9_mean_scores_tiers
  fig10_crossover_analysis
  fig11_professional_sequences
  fig12_model_robustness
)

for f in "${FIGS[@]}"; do
  cp "../figures/$f.png" "images/$f.png"
done

echo "copy-figures: synced ${#FIGS[@]} figures into website/images/"
