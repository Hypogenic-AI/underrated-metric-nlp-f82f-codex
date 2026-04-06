## Literature Review

### Research Area Overview
This project targets a specific novelty failure mode in LLMs: producing "obvious" responses for prompts that demand non-obviousness (e.g., "most underrated X"). The closest prior work spans three clusters: (1) decoding diversity and degeneration, (2) novelty/creativity-oriented generation interventions, and (3) novelty/serendipity evaluation in recommendation-style ranking.

### Key Papers

#### Paper 1: The Curious Case of Neural Text Degeneration
- **Authors**: Ari Holtzman et al.
- **Year**: 2019
- **Source**: arXiv (1904.09751)
- **Key Contribution**: Shows high-likelihood decoding produces repetitive/bland outputs; introduces nucleus sampling.
- **Methodology**: Distributional analysis of generated text under different decoding strategies.
- **Datasets Used**: Open-ended generation settings (paper-level).
- **Results**: Sampling strategy strongly affects diversity/quality tradeoff.
- **Code Available**: Not central in this repo; methods are widely reimplemented.
- **Relevance to Our Research**: Provides foundational reason why "most underrated" answers may collapse to obvious tokens under standard decoding.

#### Paper 2: Diverse Beam Search
- **Authors**: Ashwin K. Vijayakumar et al.
- **Year**: 2016
- **Source**: arXiv (1610.02424)
- **Key Contribution**: Diversity-augmented beam objective for multiple diverse candidates.
- **Methodology**: Group-wise beam search with diversity penalties.
- **Datasets Used**: Image captioning, MT, VQG.
- **Results**: Better diversity with modest compute overhead.
- **Code Available**: Various implementations in open-source.
- **Relevance to Our Research**: Candidate decoding baseline to reduce "obvious underrated" mode collapse.

#### Paper 3: MAUVE
- **Authors**: Krishna Pillutla et al.
- **Year**: 2021
- **Source**: arXiv (2102.01454)
- **Key Contribution**: Distribution-level metric comparing machine vs. human text.
- **Methodology**: Divergence frontiers in quantized embedding space.
- **Datasets Used**: Open-ended generation tasks.
- **Results**: Better alignment with human judgments than many overlap metrics.
- **Code Available**: Yes (cloned: `code/mauve/`).
- **Relevance to Our Research**: Useful secondary metric when judging novelty without sacrificing fluency.

#### Paper 4: CommonGen
- **Authors**: Bill Yuchen Lin et al.
- **Year**: 2019/2020
- **Source**: arXiv (1911.03705), EMNLP Findings
- **Key Contribution**: Benchmark for constrained generative commonsense.
- **Methodology**: Generate plausible sentences from concept sets.
- **Datasets Used**: CommonGen benchmark itself.
- **Results**: Significant human-model gap in compositional generation.
- **Code Available**: Yes (cloned: `code/CommonGen/`).
- **Relevance to Our Research**: Supplies controlled generation setting to test novelty under constraints.

#### Paper 5: Creative Beam Search: LLM-as-a-Judge
- **Authors**: Giorgio Franceschelli, Mirco Musolesi
- **Year**: 2024
- **Source**: arXiv (2405.00099)
- **Key Contribution**: Combines diverse generation with an LLM-based validation step.
- **Methodology**: Generate candidate responses, then judge/filter for creativity.
- **Datasets Used**: Qualitative evaluation setup.
- **Results**: Reported qualitative gains over standard sampling.
- **Code Available**: Not clearly linked in paper metadata.
- **Relevance to Our Research**: Directly supports a two-stage "generate + novelty-judge" design for underratedness tasks.

#### Paper 6: Multi-Novelty
- **Authors**: Arash Lagzian et al.
- **Year**: 2025
- **Source**: arXiv (2502.12700)
- **Key Contribution**: Inference-time multi-view brainstorming to increase novelty/diversity.
- **Methodology**: Prompt enrichment from textual/visual perspectives; no model retraining.
- **Datasets Used**: Mixed prompting/creative tasks across multiple LLMs.
- **Results**: Reported diversity gains (including lexical and self-BLEU-related improvements).
- **Code Available**: Not confirmed from paper metadata.
- **Relevance to Our Research**: Strong candidate intervention for generating less-obvious "underrated" outputs.

#### Paper 7: Optimizing Novelty of Top-k Recommendations using LLMs and RL
- **Authors**: Amit Sharma et al.
- **Year**: 2024
- **Source**: KDD 2024 / arXiv (2406.14169)
- **Key Contribution**: Optimizes novelty@k via RL despite non-differentiable top-k objective.
- **Methodology**: Policy-gradient formulation with item-wise reward decomposition and reduced action space.
- **Datasets Used**: Query-keyword, ORCAS, Amazon-based product recommendation.
- **Results**: Novelty gains with limited recall loss.
- **Code Available**: Not directly linked in metadata.
- **Relevance to Our Research**: Supplies formal novelty objective design relevant to "most underrated" ranking tasks.

#### Paper 8: Bursting Filter Bubble (SERAL)
- **Authors**: Yunjia Xi et al.
- **Year**: 2025
- **Source**: arXiv (2502.13539)
- **Key Contribution**: LLM-aligned serendipity recommendation framework.
- **Methodology**: Profile compression + alignment + nearline adaptation.
- **Datasets Used**: Industrial logs (Taobao deployment context).
- **Results**: Reported gains in exposure/click/transaction for serendipitous items.
- **Code Available**: Not provided in metadata.
- **Relevance to Our Research**: Practical signal for evaluating "unexpected yet relevant" recommendations.

#### Paper 9: Harnessing LLMs for Scientific Novelty Detection
- **Authors**: Yan Liu et al.
- **Year**: 2025
- **Source**: arXiv (2505.24615)
- **Key Contribution**: Benchmarks and methods for novelty detection at idea level.
- **Methodology**: LLM-assisted idea extraction + distillation to lightweight retriever.
- **Datasets Used**: New ND datasets in marketing and NLP domains.
- **Results**: Improved retrieval and novelty detection over baselines.
- **Code Available**: Link indicated in abstract metadata.
- **Relevance to Our Research**: Suggests benchmark-construction strategy for underratedness evaluation sets.

#### Paper 10: Dynamic User KG for Serendipity Recommendation
- **Authors**: Qian Yong et al.
- **Year**: 2025
- **Source**: arXiv (2508.04032)
- **Key Contribution**: LLM-based dynamic user knowledge graph for serendipitous retrieval.
- **Methodology**: Two-hop reasoning + near-line adaptation in industrial RS.
- **Datasets Used**: Industrial-scale app data.
- **Results**: Improved novelty-related online metrics.
- **Code Available**: Not identified.
- **Relevance to Our Research**: Supports user-profile-aware underratedness/novelty candidate generation.

### Deep Reading Notes (Chunk-based)
Using the PDF chunker, two key papers were read across all chunk files:
- `papers/2406.14169_optimizing_novelty_of_top_k_recommendations_using_large_lang.pdf` (4 chunks)
- `papers/2502.12700_multi_novelty_improve_the_diversity_and_novelty_of_contents_.pdf` (5 chunks)

Key extracted details:
- 2406.14169 formalizes novelty@k as distinct from diversity and addresses non-differentiable ranking via RL.
- 2406.14169 reports datasets including Query-Keyword, ORCAS, and Amazon recommendation data.
- 2502.12700 evaluates multiple diversity proxies (lexical entropy, semantic embedding spread, self-BLEU-like signals).
- 2502.12700 emphasizes inference-time methods rather than finetuning, which fits low-cost experimentation.

### Common Methodologies
- Decoding-time diversity control: Used in Diverse Beam Search, Creative Beam Search.
- Distributional evaluation: Used in MAUVE and related degeneration analyses.
- Novelty-aware ranking/recommendation: Used in novelty@k RL and serendipity recommendation papers.
- LLM-as-judge or alignment components: Used in Creative Beam Search and serendipity alignment frameworks.

### Standard Baselines
- Greedy decoding / vanilla beam search.
- Nucleus sampling (top-p) and temperature sampling.
- Diverse beam search variants.
- Relevance-only recommendation rankers (for novelty@k comparisons).

### Evaluation Metrics
- Diversity metrics: Distinct-n, self-BLEU, lexical entropy.
- Distributional metrics: MAUVE.
- Ranking novelty metrics: Novelty@k, serendipity-related CTR/exposure deltas.
- Quality safeguards: task success/relevance (to ensure novelty is not pure randomness).

### Datasets in the Literature
- CommonGen for constrained creative generation.
- ORCAS and large-scale query-recommendation corpora for novelty@k.
- Amazon-style recommendation data for relevance-vs-novelty tradeoff.
- Industrial behavior logs in serendipity deployment papers.

### Gaps and Opportunities
- Gap 1: No standard benchmark directly for prompts like "most underrated X" with ground-truth underratedness labels.
- Gap 2: Current novelty metrics can reward unusual but low-quality or irrelevant outputs.
- Gap 3: Many recent novelty papers are preprints with limited standardized reproducible code.

### Recommendations for Our Experiment
- **Recommended datasets**:
  - `datasets/common_gen` for controlled novelty in generation.
  - `datasets/movielens_small` for constructing underratedness labels via high rating + low popularity.
- **Recommended baselines**:
  - Greedy / top-p / beam search.
  - Diverse beam search and LLM-as-judge reranking.
- **Recommended metrics**:
  - Distinct-n, self-BLEU, MAUVE, and task-specific underrated-hit@k.
- **Methodological considerations**:
  - Separate novelty from relevance.
  - Report both absolute performance and novelty-adjusted performance.
  - Add human or rubric-based checks for "genuine underratedness".
