#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUBREDDITS = ["movies", "MovieSuggestions", "Letterboxd", "TrueFilm"]
TITLE_HIT_THRESHOLD = 5
SEED = 42


def normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def reddit_search_count(title: str, subreddit: str) -> int:
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    headers = {"User-Agent": "underrated-metric-research/1.0"}
    q = f'"{title}" underrated movie'
    params = {
        "q": q,
        "restrict_sr": 1,
        "sort": "relevance",
        "t": "all",
        "limit": 100,
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=8)
        if r.status_code != 200:
            return 0
        children = r.json().get("data", {}).get("children", [])
        return len(children)
    except Exception:
        return 0


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    sd = d.std(ddof=1)
    return 0.0 if sd == 0 else float(d.mean() / sd)


def permutation_paired(a: np.ndarray, b: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    diff = b - a
    obs = abs(diff.mean())
    n_perm = 10000
    c = 0
    for _ in range(n_perm):
        s = rng.choice([-1, 1], size=len(diff))
        if abs((diff * s).mean()) >= obs:
            c += 1
    return (c + 1) / (n_perm + 1)


def run_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in df.model.unique():
        sub = df[df.model == model]
        for cond in ["anti_consensus", "anti_consensus_reasoned"]:
            m = (
                sub[sub.condition == "baseline"]
                .merge(sub[sub.condition == cond], on=["model", "genre", "run_idx"], suffixes=("_base", "_alt"))
            )
            if m.empty:
                continue
            a = m.UNS_hit_base.to_numpy(float)
            b = m.UNS_hit_alt.to_numpy(float)
            ov_a = m.overlap_hit_base.to_numpy(int)
            ov_b = m.overlap_hit_alt.to_numpy(int)
            try:
                _, p_w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            except Exception:
                p_w = np.nan
            p_perm = permutation_paired(a, b)
            rows.append(
                {
                    "model": model,
                    "comparison": f"baseline_vs_{cond}",
                    "n": len(m),
                    "baseline_uns_mean": float(a.mean()),
                    "alt_uns_mean": float(b.mean()),
                    "delta_uns": float((b - a).mean()),
                    "wilcoxon_p": p_w,
                    "perm_p": p_perm,
                    "cohens_d_paired": paired_cohens_d(a, b),
                    "baseline_overlap_rate": float(ov_a.mean()),
                    "alt_overlap_rate": float(ov_b.mean()),
                    "delta_overlap": float((ov_b - ov_a).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        rej, padj, _, _ = multipletests(out.perm_p.tolist(), alpha=0.05, method="holm")
        out["perm_p_holm"] = padj
        out["significant_holm"] = rej
    return out


def main() -> None:
    df = pd.read_csv(RESULTS / "model_outputs_scored.csv")

    titles = sorted({t for t in df["parsed_title"].dropna().astype(str).tolist() if t.strip()})
    valid_titles = []
    for t in titles:
        tt = t.strip()
        if "{" in tt or "}" in tt or ":" in tt:
            continue
        if len(tt) < 2:
            continue
        valid_titles.append(tt)

    hit_rows = []
    cache_path = RESULTS / "reddit_title_hits.csv"
    cached = {}
    if cache_path.exists():
        old = pd.read_csv(cache_path)
        if "parsed_title" in old.columns and "reddit_hits_total" in old.columns:
            for row in old.itertuples(index=False):
                cached[str(row.parsed_title)] = {
                    "reddit_hits_total": int(getattr(row, "reddit_hits_total")),
                    "hits_movies": int(getattr(row, "hits_movies", 0)),
                    "hits_MovieSuggestions": int(getattr(row, "hits_MovieSuggestions", 0)),
                    "hits_Letterboxd": int(getattr(row, "hits_Letterboxd", 0)),
                    "hits_TrueFilm": int(getattr(row, "hits_TrueFilm", 0)),
                }

    for t in valid_titles:
        if t in cached:
            hit_rows.append({"parsed_title": t, **cached[t]})
            continue
        total = 0
        per_sub = {}
        for sub in SUBREDDITS:
            c = reddit_search_count(t, sub)
            per_sub[sub] = c
            total += c
        hit_rows.append({"parsed_title": t, "reddit_hits_total": total, **{f"hits_{k}": v for k, v in per_sub.items()}})

    hits = pd.DataFrame(hit_rows)
    if hits.empty:
        hits = pd.DataFrame(columns=["parsed_title", "reddit_hits_total"])
    hits.to_csv(RESULTS / "reddit_title_hits.csv", index=False)

    df = df.merge(hits[["parsed_title", "reddit_hits_total"]], on="parsed_title", how="left")
    df["reddit_hits_total"] = df["reddit_hits_total"].fillna(0)
    df["overlap_hit"] = (df["reddit_hits_total"] >= TITLE_HIT_THRESHOLD).astype(int)
    df["underrated_pct_filled"] = df["underrated_pct"].fillna(0.0)
    df["UNS_hit"] = 0.6 * df["underrated_pct_filled"] + 0.4 * (1 - df["overlap_hit"])

    summary = (
        df.groupby(["model", "condition"], as_index=False)
        .agg(
            n=("UNS_hit", "size"),
            UNS_hit_mean=("UNS_hit", "mean"),
            UNS_hit_std=("UNS_hit", "std"),
            overlap_hit_rate=("overlap_hit", "mean"),
            reddit_hits_mean=("reddit_hits_total", "mean"),
            match_rate=("matched", "mean"),
            parse_error_rate=("parse_error", lambda x: x.notna().mean()),
        )
    )
    summary.to_csv(RESULTS / "summary_metrics_reddit_hits.csv", index=False)

    stats_df = run_stats(df)
    stats_df.to_csv(RESULTS / "statistical_tests_reddit_hits.csv", index=False)

    df.to_csv(RESULTS / "model_outputs_scored_with_reddit_hits.csv", index=False)
    print("Saved reddit-hit-based summaries.")


if __name__ == "__main__":
    main()
