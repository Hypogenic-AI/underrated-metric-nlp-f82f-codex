## Motivation & Novelty Assessment

### Why This Research Matters
If LLMs cannot move beyond socially obvious answers when explicitly asked for underrated items, then their usefulness for ideation, discovery, and innovation support is limited. A robust diagnostic for this failure mode would help benchmark creativity-oriented prompting methods and model improvements. This matters for both consumer use (recommendation, discovery) and research use (idea generation), where non-obvious but relevant outputs are the goal.

### Gap in Existing Work
Based on the literature review, prior work measures diversity, fluency, and distributional novelty (e.g., Distinct-n, MAUVE, novelty@k), but there is no standard benchmark for the specific “most underrated X” behavior. Existing novelty metrics also conflate unusualness with quality/relevance, and rarely test whether outputs merely echo crowd consensus. There is a gap in measuring anti-consensus novelty while preserving plausibility.

### Our Novel Contribution
We operationalize an “Underrated Novelty Diagnostic” that jointly scores: (1) alignment with popular Reddit consensus (obviousness proxy), (2) data-grounded underratedness from MovieLens (high rating, low popularity), and (3) response quality validity checks. This creates a practical metric for whether a model can propose genuinely underrated candidates rather than repeating familiar “underrated” picks.

### Experiment Justification
- Experiment 1: Baseline prompting across real LLM APIs to measure consensus echoing. Needed to test the core claim that default LLM behavior is obvious.
- Experiment 2: Prompt interventions (anti-consensus + constrained reasoning) to test whether failure is prompt-sensitive or model-intrinsic. Needed to separate capability from instruction-following.
- Experiment 3: Robustness across genres and random seeds (temperature runs), with statistical tests and effect sizes. Needed to ensure conclusions are not due to narrow prompt/category choice.

## Research Question
Can “most underrated X” prompts reliably diagnose LLM novelty limitations by distinguishing genuine underrated discovery from obvious crowd-consensus answers?

## Background and Motivation
Literature on neural text degeneration and diversity-aware decoding suggests that standard decoding often collapses toward safe, high-probability outputs. Recommendation and serendipity work shows novelty can be optimized but requires explicit objectives and trade-off controls. This project fills a benchmark gap by testing an ecologically meaningful prompt form (“most underrated X”) with explicit anti-consensus evaluation.

## Hypothesis Decomposition
- H1 (Consensus Echo): Under baseline prompting, models disproportionately produce items that overlap with high-frequency Reddit “underrated” mentions.
- H2 (Weak True Underratedness): Baseline outputs underperform on data-grounded underratedness score (rating-popularity residual) compared with intervention prompts.
- H3 (Prompt Gain, Incomplete): Anti-consensus prompting improves novelty metrics but does not fully eliminate obviousness across categories.

Independent variables:
- Model (e.g., GPT-4.1 vs GPT-5 via API availability)
- Prompt condition (baseline, anti-consensus, anti-consensus+justification)
- Category/genre (MovieLens genres)
- Sampling seed/temperature run

Dependent variables:
- Reddit Consensus Overlap@k
- Underratedness percentile in MovieLens
- Composite Underrated Novelty Score (UNS)
- Format validity / parse success

Success criteria:
- Significant reduction in consensus overlap under intervention (p < 0.05)
- Significant increase in UNS with non-trivial effect size (|d| >= 0.3)
- Findings stable across genres and repeat runs

Alternative explanations to test:
- Prompt wording artifacts
- Model-specific catalog memorization
- Genre popularity imbalance

## Proposed Methodology

### Approach
Build a genre-specific benchmark from MovieLens and Reddit-derived consensus mentions. Query real LLMs with controlled prompts and evaluate outputs using a combined anti-consensus and underratedness metric. Compare baseline vs intervention prompts with paired statistical tests.

### Experimental Steps
1. Build movie metadata table and underratedness proxy from MovieLens (`avg_rating` high, `rating_count` low).
2. Collect Reddit consensus lists for “underrated [genre] movies” from popular subreddits (`movies`, `MovieSuggestions`, `Letterboxd`).
3. Generate model outputs with real APIs for each genre and prompt condition, repeated across seeds.
4. Normalize and match returned movie names to MovieLens titles (fuzzy and year-aware matching).
5. Compute metrics: consensus overlap, underratedness percentile, UNS, validity rate.
6. Run statistical tests and effect sizes across paired genre-condition samples.
7. Perform qualitative error analysis on high-overlap (obvious) and unmatched outputs.

### Baselines
- Baseline prompt: “What is the most underrated [genre] movie? Return one title and year.”
- Intervention A: Anti-consensus instruction (avoid commonly cited picks).
- Intervention B: Anti-consensus + short rationale tied to under-recognition evidence.
- Model baseline: at least one modern model; target two (GPT-4.1, GPT-5 or OpenRouter equivalents if needed).

### Evaluation Metrics
- Consensus Overlap@1: whether output appears in top Reddit consensus list for that genre.
- Consensus Rank Score: normalized inverse frequency/rank from Reddit mentions.
- MovieLens Underratedness Score: standardized residual from high rating minus log popularity.
- Composite UNS: weighted score favoring low consensus + high underratedness + valid match.
- Coverage: percentage of outputs matched to MovieLens catalog.

### Statistical Analysis Plan
- Null hypotheses:
  - H0a: No difference in Consensus Overlap between baseline and intervention prompts.
  - H0b: No difference in UNS between baseline and intervention prompts.
- Tests:
  - Paired permutation test (primary, distribution-robust)
  - Wilcoxon signed-rank test (secondary)
  - McNemar test for paired binary overlap (if contingency permits)
- Effect sizes:
  - Cohen’s d (paired)
  - Cliff’s delta (robust)
- Multiple comparisons:
  - Holm correction across condition pairs
- Significance:
  - alpha = 0.05

## Expected Outcomes
Supporting hypothesis: baseline condition has high consensus overlap and lower UNS; interventions improve UNS but still leave substantial overlap in some genres. Refuting hypothesis: low baseline overlap and high underratedness without intervention.

## Timeline and Milestones
- Milestone 1 (20 min): Environment/GPU verification, dependency finalization.
- Milestone 2 (35 min): Data prep (MovieLens processing + Reddit consensus extraction).
- Milestone 3 (45 min): API experiment harness implementation.
- Milestone 4 (45 min): Full experiment runs and artifact generation.
- Milestone 5 (30 min): Statistical analysis + plots + error analysis.
- Milestone 6 (25 min): REPORT.md/README.md finalization and validation.

## Potential Challenges
- API availability/model access mismatch: fallback to OpenRouter model aliases, log exact IDs.
- Reddit endpoint throttling: cache responses, include retry/backoff and timestamped snapshots.
- Title matching ambiguity: use conservative year-aware fuzzy threshold and report unmatched rate.
- Small sample per genre: bootstrap confidence intervals and permutation tests.

## Success Criteria
- Complete reproducible pipeline under `src/` with saved raw outputs and metrics.
- Statistical evidence for or against hypothesis with effect sizes and confidence intervals.
- REPORT.md contains actual tables/figures and explicit limitations.
