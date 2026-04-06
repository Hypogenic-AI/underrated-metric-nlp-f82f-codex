## Resources Catalog

### Summary
This document catalogs all resources gathered for the project "'Most Underrated' as a Novelty Metric", including papers, datasets, and code repositories.

### Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models | Vijayakumar et al. | 2016 | papers/1610.02424_diverse_beam_search_decoding_diverse_solutions_from_neural_s.pdf | Foundational diversity decoding baseline |
| The Curious Case of Neural Text Degeneration | Holtzman et al. | 2019 | papers/1904.09751_the_curious_case_of_neural_text_degeneration.pdf | Degeneration analysis; nucleus sampling |
| CommonGen: A Constrained Text Generation Challenge for Generative Commonsense Reasoning | Lin et al. | 2019/2020 | papers/1911.03705_commongen_a_constrained_text_generation_challenge_for_genera.pdf | Constrained generation benchmark |
| MAUVE: Measuring the Gap Between Neural Text and Human Text | Pillutla et al. | 2021 | papers/2102.01454_mauve_measuring_the_gap_between_neural_text_and_human_text_u.pdf | Distribution-level generation metric |
| Creative Beam Search: LLM-as-a-Judge For Improving Response Generation | Franceschelli, Musolesi | 2024 | papers/2405.00099_creative_beam_search_llm_as_a_judge_for_improving_response_g.pdf | Generate-and-judge creativity pipeline |
| Optimizing Novelty of Top-k Recommendations using LLMs and RL | Sharma et al. | 2024 | papers/2406.14169_optimizing_novelty_of_top_k_recommendations_using_large_lang.pdf | Novelty@k objective with RL |
| Multi-Novelty: Improve Diversity and Novelty via Multi-Views Brainstorming | Lagzian et al. | 2025 | papers/2502.12700_multi_novelty_improve_the_diversity_and_novelty_of_contents_.pdf | Inference-time novelty intervention |
| Bursting Filter Bubble: Enhancing Serendipity Recommendations with Aligned LLMs | Xi et al. | 2025 | papers/2502.13539_bursting_filter_bubble_enhancing_serendipity_recommendations.pdf | Industrial serendipity framework |
| Harnessing Large Language Models for Scientific Novelty Detection | Liu et al. | 2025 | papers/2505.24615_harnessing_large_language_models_for_scientific_novelty_dete.pdf | Novelty detection benchmark/method |
| Enhancing Serendipity Recommendation System by Dynamic User KGs with LLMs | Yong et al. | 2025 | papers/2508.04032_enhancing_serendipity_recommendation_system_by_constructing_.pdf | User-KG reasoning for serendipity |

See `papers/README.md` for detailed descriptions.

### Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CommonGen | Hugging Face (`common_gen`) | 67,389 / 4,018 / 1,497 | constrained generation | datasets/common_gen/ | Controlled novelty/creativity benchmark |
| MovieLens Latest Small | GroupLens | 100,836 ratings; 9,742 movies | recommendation/novelty ranking | datasets/movielens_small/ | Enables underratedness proxy (rating vs popularity) |

See `datasets/README.md` for download and loading instructions.

### Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | LLM evaluation framework | code/lm-evaluation-harness/ | Use for task orchestration and evaluation logging |
| mauve | https://github.com/krishnap25/mauve | MAUVE novelty/distribution metric | code/mauve/ | Directly useful for novelty-sensitive generation evaluation |
| CommonGen | https://github.com/INK-USC/CommonGen | CommonGen baseline/evaluation resources | code/CommonGen/ | README points to maintained CommonGen-Eval repo |

See `code/README.md` for detailed descriptions.

### Resource Gathering Notes

#### Search Strategy
- Attempted `paper-finder` diligent mode first via `.claude/skills/paper-finder/scripts/find_papers.py`.
- Local paper-finder API (`http://localhost:8000`) was unavailable, so manual fallback was used.
- Manual search used arXiv API topic queries on novelty/diversity/serendipity + foundational papers.
- Deep reading for two key papers used `pdf_chunker.py` and all generated chunks.

#### Selection Criteria
- Direct relevance to LLM novelty/diversity, decoding behavior, and serendipity.
- Mix of foundational papers and recent (2024-2025) methods.
- Preference for papers with publicly available PDFs and metrics usable in experiments.

#### Challenges Encountered
- Intermittent empty responses from arXiv metadata endpoint for specific ID queries.
- Local paper-finder backend unavailable.

#### Gaps and Workarounds
- No canonical benchmark exists for "most underrated X" prompts.
- Workaround: combine CommonGen (controlled novelty generation) with MovieLens-based underratedness proxy labels.

### Recommendations for Experiment Design

1. **Primary dataset(s)**: Start with `common_gen` for generation control; use `movielens_small` to build underratedness labels and ranking tasks.
2. **Baseline methods**: Greedy/top-p/beam, diverse beam search, and LLM-as-judge reranking.
3. **Evaluation metrics**: Distinct-n, self-BLEU, MAUVE, plus underrated-hit@k and relevance checks.
4. **Code to adapt/reuse**: Use `lm-evaluation-harness` for evaluation pipelines and `mauve` for distributional novelty scoring.
