#!/usr/bin/env python3
"""Run the 'most underrated' novelty diagnostic on real LLM APIs.

Pipeline:
1) Load MovieLens and build underratedness proxy scores.
2) Collect Reddit consensus mentions for "underrated <genre> movie".
3) Query real LLM APIs (OpenAI Responses) across models + prompt conditions.
4) Match outputs to MovieLens titles and compute novelty metrics.
5) Run statistical tests, create plots, and write report-ready artifacts.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from openai import OpenAI
from rapidfuzz import fuzz, process
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
MOVIELENS_DIR = DATASETS_DIR / "movielens_small" / "ml-latest-small"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = ROOT / "logs"

for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


TARGET_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Drama",
    "Horror",
    "Romance",
    "Sci-Fi",
    "Thriller",
]

SUBREDDITS = ["movies", "MovieSuggestions", "Letterboxd", "TrueFilm"]

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "gpt-4.1": {
        "temperature": 1.0,
        "max_output_tokens": 256,
        "reasoning": None,
    },
    "gpt-5": {
        "temperature": None,
        "max_output_tokens": 512,
        "reasoning": {"effort": "low"},
    },
}

PROMPT_CONDITIONS: Dict[str, str] = {
    "baseline": (
        "You are choosing one movie. "
        "Question: What is the most underrated {genre} movie? "
        "Return strict JSON with keys title (string), year (integer or null), rationale (string)."
    ),
    "anti_consensus": (
        "You are choosing one movie. "
        "Question: What is the most underrated {genre} movie? "
        "Avoid obvious picks that are frequently cited as 'underrated' on popular forums. "
        "Prefer a less-obvious but defensible title. "
        "Return strict JSON with keys title (string), year (integer or null), rationale (string)."
    ),
    "anti_consensus_reasoned": (
        "You are choosing one movie. "
        "Question: What is the most underrated {genre} movie? "
        "Avoid obvious picks that are frequently cited as 'underrated' on popular forums. "
        "Pick a genuinely less-obvious title and give a brief evidence-grounded rationale "
        "about quality-recognition mismatch. "
        "Return strict JSON with keys title (string), year (integer or null), rationale (string)."
    ),
}

N_RUNS_PER_CELL = 2
CONSENSUS_TOP_K = 20


@dataclass
class MovieMatch:
    movie_id: Optional[int]
    matched_title: Optional[str]
    matched_year: Optional[int]
    match_score: float


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)


def normalize_title(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_year_from_title(title: str) -> Optional[int]:
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def parse_model_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"title": None, "year": None, "rationale": None, "parse_error": "empty"}

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {
                "title": obj.get("title"),
                "year": obj.get("year"),
                "rationale": obj.get("rationale"),
                "parse_error": None,
            }
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {
                    "title": obj.get("title"),
                    "year": obj.get("year"),
                    "rationale": obj.get("rationale"),
                    "parse_error": None,
                }
        except Exception:
            pass

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0] if lines else None
    year = None
    if title:
        ym = re.search(r"\b(19\d{2}|20\d{2})\b", title)
        if ym:
            year = int(ym.group(1))
            title = re.sub(r"\b(19\d{2}|20\d{2})\b", "", title).strip(" -:()")
    return {"title": title, "year": year, "rationale": None, "parse_error": "non_json"}


def load_movielens() -> pd.DataFrame:
    movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv")

    agg = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    )
    df = movies.merge(agg, on="movieId", how="left")
    df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].median())
    df["rating_count"] = df["rating_count"].fillna(0)

    df["year"] = df["title"].apply(extract_year_from_title)
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    df["title_norm"] = df["title_clean"].map(normalize_title)

    log_pop = np.log1p(df["rating_count"])
    df["underrated_raw"] = stats.zscore(df["avg_rating"], nan_policy="omit") - stats.zscore(log_pop, nan_policy="omit")
    df["underrated_pct_global"] = df["underrated_raw"].rank(pct=True)

    df["genre_list"] = df["genres"].str.split("|")
    for g in TARGET_GENRES:
        mask = df["genre_list"].map(lambda x: g in x if isinstance(x, list) else False)
        if mask.sum() > 0:
            df.loc[mask, f"underrated_pct_{g}"] = df.loc[mask, "underrated_raw"].rank(pct=True)

    return df


def reddit_search(subreddit: str, query: str, limit: int = 100) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    headers = {"User-Agent": "underrated-metric-research/1.0"}
    params = {
        "q": query,
        "restrict_sr": 1,
        "sort": "top",
        "t": "all",
        "limit": limit,
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        return []
    payload = r.json()
    children = payload.get("data", {}).get("children", [])
    posts = []
    for ch in children:
        d = ch.get("data", {})
        posts.append(
            {
                "subreddit": subreddit,
                "title": d.get("title", ""),
                "selftext": d.get("selftext", ""),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "permalink": d.get("permalink", ""),
                "created_utc": d.get("created_utc"),
            }
        )
    return posts


def build_reddit_consensus(movies_df: pd.DataFrame) -> Tuple[Dict[str, List[Dict[str, Any]]], pd.DataFrame]:
    title_lookup = movies_df[["movieId", "title_clean", "title_norm", "year", "genres"]].copy()
    norm_to_rows = defaultdict(list)
    for row in title_lookup.itertuples(index=False):
        if not row.title_norm:
            continue
        tokens = row.title_norm.split()
        # Avoid high false positives from ambiguous single-word titles in free text.
        if len(tokens) < 2:
            continue
        if len(row.title_norm) < 6:
            continue
        if row.title_norm and len(row.title_norm) >= 4:
            norm_to_rows[row.title_norm].append(row)

    all_posts = []
    by_genre_mentions: Dict[str, Counter] = {g: Counter() for g in TARGET_GENRES}

    for genre in TARGET_GENRES:
        for sub in SUBREDDITS:
            query = f"underrated {genre} movie"
            posts = reddit_search(sub, query, limit=100)
            for p in posts:
                p["genre_query"] = genre
            all_posts.extend(posts)

            for p in posts:
                text = f"{p['title']} {p['selftext']}"
                text_norm = normalize_title(text)
                if len(text_norm) < 5:
                    continue

                for tnorm, rows in norm_to_rows.items():
                    if len(tnorm) < 4:
                        continue
                    if f" {tnorm} " in f" {text_norm} ":
                        for r in rows:
                            by_genre_mentions[genre][int(r.movieId)] += 1

    consensus: Dict[str, List[Dict[str, Any]]] = {}
    for genre, counter in by_genre_mentions.items():
        rows = []
        for mid, cnt in counter.most_common(CONSENSUS_TOP_K):
            mrow = movies_df.loc[movies_df["movieId"] == mid].iloc[0]
            rows.append(
                {
                    "movieId": int(mid),
                    "title": mrow["title_clean"],
                    "year": int(mrow["year"]) if not pd.isna(mrow["year"]) else None,
                    "count": int(cnt),
                }
            )
        consensus[genre] = rows

    posts_df = pd.DataFrame(all_posts)
    return consensus, posts_df


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
)
def call_model(client: OpenAI, model: str, prompt: str, run_idx: int) -> Dict[str, Any]:
    spec = MODEL_SPECS[model]
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": prompt + f"\nRun identifier: {run_idx}",
        "max_output_tokens": spec["max_output_tokens"],
    }
    if spec.get("reasoning"):
        kwargs["reasoning"] = spec["reasoning"]

    if spec.get("temperature") is not None:
        kwargs["temperature"] = spec["temperature"]

    try:
        response = client.responses.create(**kwargs, timeout=45)
    except Exception:
        if "temperature" in kwargs:
            kwargs.pop("temperature")
            response = client.responses.create(**kwargs, timeout=45)
        else:
            raise

    out_text = getattr(response, "output_text", "") or ""
    if not out_text:
        try:
            fragments = []
            for item in response.output:
                if getattr(item, "type", None) == "message" and getattr(item, "content", None):
                    for content in item.content:
                        if getattr(content, "type", None) in {"output_text", "text"}:
                            txt = getattr(content, "text", "")
                            if txt:
                                fragments.append(txt)
            out_text = "\n".join(fragments)
        except Exception:
            out_text = ""

    return {
        "response_id": response.id,
        "status": getattr(response, "status", None),
        "text": out_text,
        "usage": response.usage.model_dump() if hasattr(response.usage, "model_dump") else None,
    }


def build_genre_movie_index(movies_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    genre_map = {}
    for g in TARGET_GENRES:
        mask = movies_df["genre_list"].map(lambda x: g in x if isinstance(x, list) else False)
        genre_map[g] = movies_df[mask].copy()
    return genre_map


def match_movie(candidate_title: Optional[str], candidate_year: Optional[int], genre_df: pd.DataFrame) -> MovieMatch:
    if not candidate_title or not isinstance(candidate_title, str):
        return MovieMatch(None, None, None, 0.0)

    q = normalize_title(candidate_title)
    if not q:
        return MovieMatch(None, None, None, 0.0)

    choices = genre_df["title_norm"].tolist()
    if not choices:
        return MovieMatch(None, None, None, 0.0)

    best = process.extractOne(q, choices, scorer=fuzz.WRatio)
    if best is None:
        return MovieMatch(None, None, None, 0.0)

    best_norm, score, idx = best
    row = genre_df.iloc[idx]

    if score < 75:
        return MovieMatch(None, None, None, float(score))

    matched_year = int(row["year"]) if not pd.isna(row["year"]) else None
    if candidate_year and matched_year and abs(candidate_year - matched_year) > 2:
        return MovieMatch(None, None, None, float(score))

    return MovieMatch(int(row["movieId"]), row["title_clean"], matched_year, float(score))


def consensus_rank(movie_id: Optional[int], consensus_rows: List[Dict[str, Any]]) -> Optional[int]:
    if movie_id is None:
        return None
    for i, row in enumerate(consensus_rows, start=1):
        if row["movieId"] == movie_id:
            return i
    return None


def consensus_rank_by_title(candidate_title: Optional[str], consensus_rows: List[Dict[str, Any]]) -> Optional[int]:
    if not candidate_title:
        return None
    q = normalize_title(candidate_title)
    if not q:
        return None
    titles = [normalize_title(row["title"]) for row in consensus_rows]
    if not titles:
        return None
    best = process.extractOne(q, titles, scorer=fuzz.WRatio)
    if best is None:
        return None
    _, score, idx = best
    if score < 88:
        return None
    return idx + 1


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def compute_uns(underrated_pct: Optional[float], c_rank: Optional[int], k: int) -> float:
    u = underrated_pct if underrated_pct is not None else 0.0
    if c_rank is None:
        consensus_novel = 1.0
    else:
        consensus_penalty = 1.0 - ((c_rank - 1) / max(1, k - 1))
        consensus_novel = 1.0 - consensus_penalty
    return 0.6 * u + 0.4 * consensus_novel


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = b - a
    s = diff.std(ddof=1)
    if s == 0:
        return 0.0
    return float(diff.mean() / s)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    return float((gt - lt) / n) if n else 0.0


def permutation_paired(a: np.ndarray, b: np.ndarray) -> float:
    diffs = b - a
    observed = abs(diffs.mean())
    rng = np.random.default_rng(RANDOM_SEED)
    count = 0
    n_perm = 10000
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        stat = abs((diffs * signs).mean())
        if stat >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def run_stats(eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    binary_records = []

    for model in eval_df["model"].unique():
        sub = eval_df[eval_df["model"] == model].copy()
        for cond in ["anti_consensus", "anti_consensus_reasoned"]:
            merged = (
                sub[sub["condition"] == "baseline"]
                .merge(
                    sub[sub["condition"] == cond],
                    on=["genre", "run_idx", "model"],
                    suffixes=("_base", "_alt"),
                )
            )
            if merged.empty:
                continue

            a_uns = merged["UNS_base"].to_numpy(dtype=float)
            b_uns = merged["UNS_alt"].to_numpy(dtype=float)
            a_ov = merged["consensus_overlap_base"].to_numpy(dtype=int)
            b_ov = merged["consensus_overlap_alt"].to_numpy(dtype=int)

            try:
                w_stat, w_p = stats.wilcoxon(a_uns, b_uns, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = np.nan, np.nan

            try:
                table = np.array(
                    [
                        [np.sum((a_ov == 1) & (b_ov == 1)), np.sum((a_ov == 1) & (b_ov == 0))],
                        [np.sum((a_ov == 0) & (b_ov == 1)), np.sum((a_ov == 0) & (b_ov == 0))],
                    ]
                )
                mcn = mcnemar(table, exact=False, correction=True)
                mcn_p = mcn.pvalue
            except Exception:
                mcn_p = np.nan

            p_perm = permutation_paired(a_uns, b_uns)
            d = paired_cohens_d(a_uns, b_uns)
            cd = cliffs_delta(b_uns, a_uns)

            records.append(
                {
                    "model": model,
                    "comparison": f"baseline_vs_{cond}",
                    "n": len(merged),
                    "baseline_uns_mean": float(np.mean(a_uns)),
                    "alt_uns_mean": float(np.mean(b_uns)),
                    "delta_uns": float(np.mean(b_uns - a_uns)),
                    "wilcoxon_stat": safe_float(w_stat),
                    "wilcoxon_p": safe_float(w_p),
                    "perm_p": p_perm,
                    "cohens_d_paired": d,
                    "cliffs_delta": cd,
                    "baseline_overlap_rate": float(np.mean(a_ov)),
                    "alt_overlap_rate": float(np.mean(b_ov)),
                    "delta_overlap": float(np.mean(b_ov - a_ov)),
                    "mcnemar_p": safe_float(mcn_p),
                }
            )

    stats_df = pd.DataFrame(records)
    if not stats_df.empty:
        pvals = stats_df["perm_p"].tolist()
        rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        stats_df["perm_p_holm"] = p_adj
        stats_df["significant_holm"] = rej

    return stats_df, pd.DataFrame(binary_records)


def summarize_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        eval_df.groupby(["model", "condition"], as_index=False)
        .agg(
            n=("UNS", "size"),
            UNS_mean=("UNS", "mean"),
            UNS_std=("UNS", "std"),
            overlap_rate=("consensus_overlap", "mean"),
            match_rate=("matched", "mean"),
            underrated_pct_mean=("underrated_pct", "mean"),
            parse_error_rate=("parse_error_flag", "mean"),
        )
        .sort_values(["model", "condition"])
    )
    return out


def plot_results(eval_df: pd.DataFrame) -> None:
    set_plot_style()

    plt.figure()
    sns.barplot(data=eval_df, x="condition", y="UNS", hue="model", errorbar=("ci", 95))
    plt.title("Composite Underrated Novelty Score (UNS) by Condition")
    plt.xlabel("Prompt Condition")
    plt.ylabel("UNS (0-1)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "uns_by_condition_model.png", dpi=180)
    plt.close()

    plt.figure()
    sns.barplot(data=eval_df, x="condition", y="consensus_overlap", hue="model", errorbar=("ci", 95))
    plt.title("Consensus Overlap Rate by Condition")
    plt.xlabel("Prompt Condition")
    plt.ylabel("Overlap with Reddit Consensus (0/1)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "consensus_overlap_by_condition_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=eval_df, x="genre", y="UNS", hue="condition")
    plt.title("UNS Distribution by Genre and Prompt Condition")
    plt.xlabel("Genre")
    plt.ylabel("UNS")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "uns_by_genre_condition_boxplot.png", dpi=180)
    plt.close()


def main() -> None:
    start = time.time()

    env_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": os.sys.version,
        "seed": RANDOM_SEED,
        "models": list(MODEL_SPECS.keys()),
        "n_runs_per_cell": N_RUNS_PER_CELL,
        "target_genres": TARGET_GENRES,
        "subreddits": SUBREDDITS,
    }

    (RESULTS_DIR / "config.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    movies_df = load_movielens()
    movies_df.to_csv(RESULTS_DIR / "movielens_with_underrated_scores.csv", index=False)

    consensus_cache = RESULTS_DIR / "reddit_consensus.json"
    posts_cache = RESULTS_DIR / "reddit_posts.csv"

    if consensus_cache.exists() and posts_cache.exists():
        consensus = json.loads(consensus_cache.read_text(encoding="utf-8"))
        posts_df = pd.read_csv(posts_cache)
    else:
        consensus, posts_df = build_reddit_consensus(movies_df)
        consensus_cache.write_text(json.dumps(consensus, indent=2), encoding="utf-8")
        posts_df.to_csv(posts_cache, index=False)

    genre_movie_index = build_genre_movie_index(movies_df)

    client = OpenAI(timeout=45, max_retries=2)
    rows = []
    checkpoint_path = RESULTS_DIR / "model_outputs_partial.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    for model in MODEL_SPECS:
        for genre in TARGET_GENRES:
            gdf = genre_movie_index[genre]
            consensus_rows = consensus.get(genre, [])
            for cond, template in PROMPT_CONDITIONS.items():
                for run_idx in range(N_RUNS_PER_CELL):
                    prompt = template.format(genre=genre)
                    try:
                        result = call_model(client, model=model, prompt=prompt, run_idx=run_idx)
                    except Exception as exc:
                        result = {
                            "response_id": None,
                            "status": "error",
                            "text": "",
                            "usage": None,
                            "error": str(exc),
                        }
                    parsed = parse_model_json(result["text"])

                    ctitle = parsed.get("title")
                    cyear = parsed.get("year")
                    try:
                        cyear = int(cyear) if cyear is not None else None
                    except Exception:
                        cyear = None

                    mm = match_movie(ctitle, cyear, gdf)
                    c_rank = consensus_rank(mm.movie_id, consensus_rows)
                    if c_rank is None:
                        c_rank = consensus_rank_by_title(ctitle, consensus_rows)

                    if mm.movie_id is None:
                        underrated_pct = None
                    else:
                        mrow = gdf[gdf["movieId"] == mm.movie_id].iloc[0]
                        col = f"underrated_pct_{genre}"
                        if col in mrow and not pd.isna(mrow[col]):
                            underrated_pct = float(mrow[col])
                        else:
                            underrated_pct = float(mrow["underrated_pct_global"])

                    overlap = 1 if c_rank is not None and c_rank <= CONSENSUS_TOP_K else 0
                    uns = compute_uns(underrated_pct, c_rank, CONSENSUS_TOP_K)

                    rows.append(
                        {
                            "model": model,
                            "genre": genre,
                            "condition": cond,
                            "run_idx": run_idx,
                            "prompt": prompt,
                            "response_id": result["response_id"],
                            "response_status": result["status"],
                            "raw_text": result["text"],
                            "parsed_title": ctitle,
                            "parsed_year": cyear,
                            "parsed_rationale": parsed.get("rationale"),
                            "parse_error": parsed.get("parse_error"),
                            "matched": int(mm.movie_id is not None),
                            "matched_movieId": mm.movie_id,
                            "matched_title": mm.matched_title,
                            "matched_year": mm.matched_year,
                            "match_score": mm.match_score,
                            "consensus_rank": c_rank,
                            "consensus_overlap": overlap,
                            "underrated_pct": underrated_pct,
                            "UNS": uns,
                            "usage": json.dumps(result["usage"]) if result["usage"] else None,
                            "api_error": result.get("error"),
                        }
                    )

                    # Periodic checkpoint to avoid losing progress on long API runs.
                    if len(rows) % 20 == 0:
                        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                        print(f"Checkpoint rows={len(rows)}")

    eval_df = pd.DataFrame(rows)
    eval_df["parse_error_flag"] = eval_df["parse_error"].notna().astype(int)
    eval_df.to_csv(RESULTS_DIR / "model_outputs_scored.csv", index=False)

    summary_df = summarize_metrics(eval_df)
    summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    stats_df, _ = run_stats(eval_df)
    stats_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    # Data quality snapshot
    dqc = {
        "n_movies": int(len(movies_df)),
        "n_ratings_rows": int(pd.read_csv(MOVIELENS_DIR / "ratings.csv").shape[0]),
        "missing_avg_rating": float(movies_df["avg_rating"].isna().mean()),
        "missing_rating_count": float(movies_df["rating_count"].isna().mean()),
        "duplicate_movieid": int(movies_df["movieId"].duplicated().sum()),
        "reddit_posts_collected": int(len(posts_df)),
        "runtime_seconds": float(time.time() - start),
    }
    (RESULTS_DIR / "data_quality.json").write_text(json.dumps(dqc, indent=2), encoding="utf-8")

    plot_results(eval_df)

    # Save top consensus table for report readability.
    consensus_rows = []
    for genre, items in consensus.items():
        for i, row in enumerate(items, start=1):
            consensus_rows.append({"genre": genre, "rank": i, **row})
    pd.DataFrame(consensus_rows).to_csv(RESULTS_DIR / "reddit_consensus_topk.csv", index=False)

    # Save a few representative examples.
    examples = eval_df.sample(n=min(12, len(eval_df)), random_state=RANDOM_SEED)
    examples.to_csv(RESULTS_DIR / "sample_outputs.csv", index=False)

    print("Completed experiment run.")
    print(f"Rows: {len(eval_df)}")
    print(f"Saved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
