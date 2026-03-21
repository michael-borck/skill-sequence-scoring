# Skill, Sequence, and Scoring

**A Mathematical Comparison of Traditional and World Bowling Scoring Systems**

Research project and paper for the *Journal of Quantitative Analysis in Sports* (JQAS).

## Overview

This project provides exact combinatorial analysis of both traditional ten-pin bowling scoring and the World Bowling current-frame scoring system, comparing them across four dimensions:

1. **Score distributions** — exact enumeration via dynamic programming, verified against Balmoral Software's published tables
2. **Sequence sensitivity** — traditional scoring is order-dependent; World Bowling is commutative (proved formally)
3. **Reward gradient** — traditional scoring compounds consecutive strikes super-linearly; World Bowling is flat
4. **Skill discrimination** — traditional scoring provides finer separation at elite score levels

## Key Results

| Statistic | Traditional | World Bowling |
|-----------|-------------|---------------|
| Total games | 5.73 × 10¹⁸ | 1.57 × 10¹⁸ |
| Mode | 77 | 72 |
| Mean | 79.7 | 76.5 |
| Std deviation | 13.7 | 15.2 |

World Bowling scores 290–299 are mathematically impossible.

## Project Structure

```
├── data/           # Score distribution tables (CSV)
├── src/            # Python source (distributions, analysis)
├── figures/        # Generated plots and visualisations
├── paper/          # Manuscript drafts
├── references/     # Bibliography and search notes
├── tests/          # Unit tests
├── notebooks/      # Exploratory analysis
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Compute exact score distributions
python src/bowling_distributions.py

# Run extended analysis (sequence sensitivity, reward gradient, entropy)
cd src && python bowling_analysis.py
```

## Author

Michael Borck — School of Marketing and Management, Curtin University, Perth, Australia
