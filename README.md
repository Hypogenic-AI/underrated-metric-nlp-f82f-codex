# Most Underrated as a Novelty Metric

This project tests whether LLMs give genuinely novel answers to prompts like “most underrated X,” or mostly repeat socially obvious choices.

## Key Findings
- `gpt-4.1` baseline had higher subreddit-overlap than anti-consensus prompts in robustness analysis (38.9% vs 22.2% / 11.1%).
- `gpt-4.1` anti-consensus-reasoned had highest composite robustness score (`UNS_hit=0.669`), but gains were not significant after Holm correction.
- `gpt-5` anti-consensus variants had high parse-failure rates in this setup, limiting interpretability.
- Conclusion: directional support for the hypothesis, but no corrected-significance confirmation in this pilot sample.

## Reproduce
1. Activate environment:
```bash
source .venv/bin/activate
```
2. Run main experiment:
```bash
python src/run_underrated_experiment.py
```
3. Run Reddit-hit robustness recomputation:
```bash
python src/recompute_reddit_hit_metrics.py
```

## File Structure
- `planning.md`: motivation, novelty, and experimental plan.
- `REPORT.md`: full research report.
- `src/run_underrated_experiment.py`: end-to-end pipeline.
- `src/recompute_reddit_hit_metrics.py`: robustness analysis using direct Reddit title-hit counts.
- `results/`: metrics tables, scored outputs, plots, and data quality snapshots.

See `REPORT.md` for methodology, statistics, and limitations.
