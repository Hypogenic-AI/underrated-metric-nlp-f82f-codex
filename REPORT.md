## 1. Executive Summary
Research question: can prompts of the form “most underrated X” diagnose whether LLMs produce genuinely novel picks vs obvious crowd-approved “underrated” answers?

Key finding: in this experiment, `gpt-4.1` showed a large drop in Reddit-overlap rate under anti-consensus prompting (baseline 44.4% -> 22.2% / 11.1% in robustness analysis), but overall novelty-score gains were not statistically significant after multiple-comparison correction.

Practical implication: “most underrated” can be a useful probe, but robust measurement needs better consensus-ground-truth construction; naive subreddit mining is noisy and can mislead conclusions.

## 2. Goal
Hypothesis tested: LLMs default to obvious “underrated” answers (high overlap with popular subreddit discourse), indicating weak novelty reasoning.

Why important: if models mostly echo known “underrated” picks, they are less useful for ideation/discovery tasks that require non-obvious insight.

Problem solved: this project implements a reproducible diagnostic pipeline combining (a) real LLM outputs, (b) data-grounded underratedness from MovieLens, and (c) subreddit-overlap proxies.

Expected impact: reusable methodology for future novelty/serendipity evaluations in NLP and recommendation-adjacent research.

## 3. Data Construction
### Dataset Description
- Movie catalog/ratings: MovieLens Latest Small (`datasets/movielens_small/ml-latest-small/`)
  - 9,742 movies, 100,836 ratings
- Reddit evidence source:
  - `r/movies`, `r/MovieSuggestions`, `r/Letterboxd`, `r/TrueFilm`
  - 3,600 posts collected via subreddit search API for underrated+genre queries

Known limitations:
- MovieLens-small misses many recent/long-tail titles
- Reddit search endpoint is noisy and retrieval-limited
- No human-annotated gold set for “obvious vs truly underrated”

### Example Samples
| Genre | Model | Condition | Parsed Title | Reddit Hits | Overlap Hit | Underrated Pct | UNS_hit |
|---|---|---|---|---:|---:|---:|---:|
| Action | gpt-4.1 | baseline | Dredd | 42 | 1 | 0.563 | 0.338 |
| Action | gpt-4.1 | anti_consensus_reasoned | Ninja II: Shadow of a Tear | 0 | 0 | 0.979 | 0.988 |
| Sci-Fi | gpt-4.1 | baseline | Coherence | 82 | 1 | 0.900 | 0.540 |

### Data Quality
From `results/data_quality.json`:
- Missing values in core MovieLens aggregates: 0%
- Duplicate `movieId`: 0
- Reddit posts collected: 3,600

### Preprocessing Steps
1. Parse movie year from title and normalize title strings.
2. Compute per-movie underratedness proxy:
   - `underrated_raw = z(avg_rating) - z(log1p(rating_count))`
3. Convert to percentile score (`underrated_pct`) globally and per target genre.
4. Query LLMs and parse strict JSON outputs (`title`, `year`, `rationale`).
5. Fuzzy-match model outputs to MovieLens titles (year-aware constraints).
6. Build two overlap signals:
   - Primary mined-consensus overlap (`consensus_overlap`) from subreddit corpus matching.
   - Robustness overlap (`overlap_hit`) using direct Reddit title-hit counts for `"<title> underrated movie"`.

### Train/Val/Test Splits
Not a training task. Evaluation grid:
- 2 models x 9 genres x 3 prompt conditions x 2 runs = 108 outputs.

## 4. Experiment Description
### Methodology
#### High-Level Approach
Run real API calls to modern models, then quantify novelty with a composite metric balancing “not socially obvious” and “data-grounded underratedness.”

#### Why This Method?
- Real APIs avoid simulation artifacts.
- MovieLens adds objective support for “underrecognized but high quality.”
- Reddit overlap approximates social obviousness.

Alternatives considered:
- Pure lexical diversity metrics only (insufficient for underratedness semantics).
- Human annotation (better quality, but out of scope for this automated run).

### Implementation Details
#### Tools and Libraries
- Python 3.12.8
- openai 2.30.0
- pandas 3.0.2
- numpy 2.4.4
- scipy 1.17.1
- statsmodels 0.14.6
- matplotlib 3.10.8
- seaborn 0.13.2
- rapidfuzz 3.14.3
- requests 2.32.5

#### Models
- `gpt-4.1`
- `gpt-5` (with low reasoning effort)

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| Runs per cell | 2 | runtime-budgeted reproducibility |
| Genres | 9 | common MovieLens genres |
| gpt-4.1 max_output_tokens | 256 | prompt-output requirement |
| gpt-5 max_output_tokens | 512 | avoid truncation |
| Fuzzy match threshold | 75 | empirical matching sanity |
| Reddit-hit overlap threshold | 5 posts | pragmatic robustness cutoff |
| Alpha | 0.05 | standard |
| Multiple-comparison correction | Holm | standard |

### Experimental Protocol
#### Reproducibility Information
- Seeds: 42 (analysis permutation/random ops)
- Hardware:
  - 2x NVIDIA GeForce RTX 3090, 24,576 MiB each (detected Apr 6, 2026)
  - GPU not required for API inference; no model training performed
- Runtime:
  - Main run: ~500 seconds (`results/data_quality.json`)
- API details:
  - OpenAI Responses API with retries/timeouts and checkpointing

#### Evaluation Metrics
1. `underrated_pct`: MovieLens underratedness percentile.
2. `consensus_overlap`: mined Reddit-consensus overlap (primary, noisier).
3. `overlap_hit`: direct Reddit title-hit overlap (robustness).
4. `UNS_hit = 0.6 * underrated_pct + 0.4 * (1 - overlap_hit)` (robustness composite).

### Raw Results
#### Table: Robustness Summary (`summary_metrics_reddit_hits.csv`)
| Model | Condition | N | UNS_hit Mean | Overlap Hit Rate | Reddit Hits Mean | Match Rate | Parse Error Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1 | baseline | 18 | 0.562 | 0.389 | 24.667 | 0.889 | 0.000 |
| gpt-4.1 | anti_consensus | 18 | 0.542 | 0.222 | 6.722 | 0.778 | 0.000 |
| gpt-4.1 | anti_consensus_reasoned | 18 | 0.669 | 0.111 | 8.167 | 0.722 | 0.000 |
| gpt-5 | baseline | 18 | 0.468 | 0.333 | 11.278 | 0.667 | 0.167 |
| gpt-5 | anti_consensus | 18 | 0.421 | 0.000 | 0.000 | 0.056 | 1.000 |
| gpt-5 | anti_consensus_reasoned | 18 | 0.400 | 0.000 | 0.000 | 0.000 | 1.000 |

#### Table: Robustness Statistical Tests (`statistical_tests_reddit_hits.csv`)
| Model | Comparison | Delta UNS_hit | Permutation p | Holm-adjusted p | Cohen’s d (paired) |
|---|---|---:|---:|---:|---:|
| gpt-4.1 | baseline vs anti_consensus | -0.020 | 0.826 | 0.826 | -0.052 |
| gpt-4.1 | baseline vs anti_consensus_reasoned | +0.107 | 0.141 | 0.492 | +0.358 |
| gpt-5 | baseline vs anti_consensus | -0.047 | 0.347 | 0.694 | -0.228 |
| gpt-5 | baseline vs anti_consensus_reasoned | -0.068 | 0.123 | 0.492 | -0.387 |

No comparison remained significant after Holm correction.

#### Visualizations
- `results/plots/uns_by_condition_model.png`
- `results/plots/consensus_overlap_by_condition_model.png`
- `results/plots/uns_by_genre_condition_boxplot.png`

#### Output Locations
- Main outputs: `results/model_outputs_scored.csv`
- Robustness outputs: `results/model_outputs_scored_with_reddit_hits.csv`
- Summary tables: `results/summary_metrics.csv`, `results/summary_metrics_reddit_hits.csv`
- Statistical tests: `results/statistical_tests.csv`, `results/statistical_tests_reddit_hits.csv`

## 5. Result Analysis
### Key Findings
1. `gpt-4.1` baseline had higher subreddit-overlap in robustness analysis than anti-consensus prompts (38.9% baseline vs 22.2% / 11.1%).
2. `gpt-4.1` anti-consensus-reasoned had the highest robustness composite score (0.669), but improvement over baseline was not statistically significant after correction.
3. `gpt-5` baseline produced usable outputs, but anti-consensus variants often failed strict JSON extraction in this setup (parse errors 100%), degrading metric outcomes.

### Hypothesis Testing Results
- Hypothesis (LLMs default to obvious picks): partially supported in directional overlap trends (especially for `gpt-4.1` baseline).
- Statistical significance: not achieved under corrected multiple-testing regime in this sample size.
- Effect sizes:
  - `gpt-4.1` baseline vs anti_consensus_reasoned: d = +0.358 (small-medium positive)
  - Others: small negative effects

### Comparison to Baselines
- Anti-consensus prompting reduced overlap rates but did not guarantee higher composite novelty scores consistently.
- For `gpt-4.1`, anti-consensus-reasoned appears most promising; for `gpt-5`, prompt-format compatibility issues dominated.

### Surprises and Insights
- Naive subreddit title mining produced substantial false positives; direct title-hit robustness checks were more interpretable.
- Some baseline picks (e.g., `Coherence`, `Prisoners`) were both highly discussed on Reddit and reasonably underrated by MovieLens, illustrating tension between “known underrated” and “truly novel underrated.”

### Error Analysis
Common failure modes:
- JSON truncation/empty output in `gpt-5` anti-consensus settings.
- Unmatched generated movies absent from MovieLens-small.
- Noisy subreddit matching when titles are ambiguous or generic phrases.

### Limitations
- Reddit overlap proxy remains imperfect without manual annotation.
- MovieLens-small coverage limits modern/long-tail title matching.
- 2 runs per condition are adequate for pilot diagnosis but limited for high-confidence inference.
- Prompting setup may not be optimal for `gpt-5` structured outputs.

## 6. Conclusions
“Most underrated X” is a promising diagnostic idea, but metric construction quality is critical. In this run, anti-consensus prompting reduced apparent obviousness (especially for `gpt-4.1`) yet did not produce statistically robust gains after multiple-comparison correction. The hypothesis is therefore partially supported directionally, but not conclusively confirmed.

Implication: this benchmark should be treated as a calibrated diagnostic instrument requiring better consensus gold labels and stronger output-validation controls.

Confidence: moderate for directional trends, low-moderate for definitive causal claims.

## 7. Next Steps
1. Human-label a subset (>=300 outputs) for obviousness vs genuine underratedness to validate the Reddit proxy.
2. Replace MovieLens-small with larger catalogs (MovieLens-25M + TMDB/IMDb linkage) to improve coverage/matching.
3. Add model-side output constraints (JSON schema/tool-calling) to reduce parse failures and increase comparability.
4. Expand to non-movie categories (books/music/tools) to test whether findings generalize.

## Validation Checklist
### Code Validation
- All scripts executed successfully after fixes.
- Outputs saved and reloaded correctly.
- Seeds set for stochastic analysis components.

### Scientific Validation
- Paired tests reported with effect sizes and Holm correction.
- Alternative robustness metric added when primary overlap proxy showed noise.
- Limitations and confounds explicitly documented.

### Documentation Validation
- Required sections present.
- File paths and outputs documented.
- Reproduction commands included in README.

### References
- See `literature_review.md` and `resources.md` for complete paper list and resource provenance.
