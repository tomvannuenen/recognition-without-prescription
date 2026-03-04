#!/usr/bin/env python3
"""
Permission-Granting Speech Act Analysis

Analyzes permission-granting markers in LLM vs human advice, following
Stevanovic & Peräkylä's (2012) framework of deontic authority transfer
and Palmer's (2001) taxonomy of deontic modality.

Three syntactic classes:
1. Modal permission: "you can leave", "you may say no"
2. Negated obligation: "you don't have to", "you don't owe"
3. Entitlement assertions: "you deserve", "you have the right", "you're allowed"

Usage:
    python permission_granting_analysis.py

Output: data/permission_analysis/permission_metrics.parquet
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
ADVICE_DIR = DATA_DIR / "llm_advice"
OUTPUT_DIR = DATA_DIR / "permission_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PERMISSION-GRANTING LEXICONS
# Based on Palmer (2001) deontic modality taxonomy and Stevanovic & Peräkylä (2012)
# ============================================================================

# Category 1: MODAL PERMISSION
# Grants capability/possibility to the advice-seeker
# Pattern: "you can/may + [action verb]"
MODAL_PERMISSION_PATTERNS = [
    r"\byou can leave\b",
    r"\byou can go\b",
    r"\byou can walk away\b",
    r"\byou can say no\b",
    r"\byou can refuse\b",
    r"\byou can end\b",
    r"\byou can break up\b",
    r"\byou can choose\b",
    r"\byou can decide\b",
    r"\byou can set\b",  # boundaries
    r"\byou can stop\b",
    r"\byou can cut\b",  # contact
    r"\byou may leave\b",
    r"\byou may say no\b",
    r"\byou may refuse\b",
    r"\byou're able to\b",
    r"\byou are able to\b",
    r"\byou're free to\b",
    r"\byou are free to\b",
]

# Category 2: NEGATED OBLIGATION
# Removes perceived duty/obligation
# Pattern: "you don't have to/need to/owe"
NEGATED_OBLIGATION_PATTERNS = [
    r"\byou don't have to\b",
    r"\byou do not have to\b",
    r"\byou don't need to\b",
    r"\byou do not need to\b",
    r"\byou don't owe\b",
    r"\byou do not owe\b",
    r"\byou're not obligated\b",
    r"\byou are not obligated\b",
    r"\byou're not required\b",
    r"\byou are not required\b",
    r"\byou aren't required\b",
    r"\byou aren't obligated\b",
    r"\byou're under no obligation\b",
    r"\byou have no obligation\b",
    r"\byou shouldn't have to\b",
    r"\byou should not have to\b",
    r"\bno one is forcing you\b",
    r"\bno one can force you\b",
    r"\bno one can make you\b",
    r"\byou never have to\b",
    r"\byou never owe\b",
]

# Category 3: ENTITLEMENT ASSERTIONS
# Asserts positive rights/deserving
# Pattern: "you deserve/have the right/are allowed/are entitled"
ENTITLEMENT_PATTERNS = [
    r"\byou deserve\b",
    r"\byou deserved\b",
    r"\byou have the right\b",
    r"\byou have every right\b",
    r"\byou have a right\b",
    r"\byou're allowed\b",
    r"\byou are allowed\b",
    r"\byou're entitled\b",
    r"\byou are entitled\b",
    r"\byou're permitted\b",
    r"\byou are permitted\b",
    r"\bit's your right\b",
    r"\bthat's your right\b",
    r"\byou have rights\b",
    r"\byou're worthy\b",
    r"\byou are worthy\b",
    r"\byou're worth\b",  # "you're worth more"
    r"\byou are worth\b",
    r"\byou matter\b",
]


def count_patterns(text: str, patterns: list) -> int:
    """Count total matches for a list of regex patterns."""
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower))
    return count


def extract_pattern_matches(text: str, patterns: list) -> list:
    """Extract all matching phrases for inspection."""
    text_lower = text.lower()
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text_lower)
        matches.extend(found)
    return matches


def compute_permission_metrics(text: str) -> dict:
    """Compute permission-granting metrics for a single text."""
    if not text or not isinstance(text, str) or len(text.strip()) < 50:
        return {"valid": False}

    n_chars = len(text)

    # Count each category
    modal_count = count_patterns(text, MODAL_PERMISSION_PATTERNS)
    negated_count = count_patterns(text, NEGATED_OBLIGATION_PATTERNS)
    entitlement_count = count_patterns(text, ENTITLEMENT_PATTERNS)

    total_permission = modal_count + negated_count + entitlement_count

    # Normalize per 1000 characters (consistent with existing metrics)
    modal_per_1k = (modal_count / n_chars) * 1000
    negated_per_1k = (negated_count / n_chars) * 1000
    entitlement_per_1k = (entitlement_count / n_chars) * 1000
    total_per_1k = (total_permission / n_chars) * 1000

    return {
        "valid": True,
        "n_chars": n_chars,
        "modal_permission_count": modal_count,
        "negated_obligation_count": negated_count,
        "entitlement_count": entitlement_count,
        "total_permission_count": total_permission,
        "modal_permission_per_1k": modal_per_1k,
        "negated_obligation_per_1k": negated_per_1k,
        "entitlement_per_1k": entitlement_per_1k,
        "total_permission_per_1k": total_per_1k,
    }


def load_human_comments() -> pd.DataFrame:
    """Load top human comment for each eligible post."""
    comments_df = pd.read_parquet(DATA_DIR / "r_relationship_advice_comments_cleaned.parquet")
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)

    # Quality filter
    quality = comments_df[
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= 5) &
        (comments_df['body'].str.len() >= 200)
    ]

    # Get top comment per post
    top_comments = quality.sort_values('score', ascending=False).groupby('post_id').first()
    top_comments = top_comments[['body', 'score']].reset_index()
    top_comments.columns = ['post_id', 'human_response', 'human_score']

    return top_comments


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0


def main():
    print("=" * 70)
    print("PERMISSION-GRANTING SPEECH ACT ANALYSIS")
    print("=" * 70)

    # Load LLM advice
    print("\nLoading LLM advice data...")
    merged_files = sorted(ADVICE_DIR.glob('llm_advice_*.parquet'))
    advice_df = pd.read_parquet(merged_files[-1])
    print(f"  Posts: {len(advice_df)}")

    # Load human comments
    print("Loading human comments...")
    human_df = load_human_comments()
    print(f"  Human comments: {len(human_df)}")

    # Merge
    df = advice_df.merge(human_df, on='post_id', how='inner')
    print(f"  Matched posts: {len(df)}")

    # Column mapping
    model_key_to_short = {
        'google_gemini-2-5-flash-lite': 'gemini',
        'deepseek_deepseek-chat-v3-0324': 'deepseek',
        'mistralai_ministral-8b': 'ministral',
        'openai_gpt-4-1-nano': 'gpt_nano',
    }

    # Compute metrics
    print("\nComputing permission metrics...")
    all_metrics = []

    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(df)}...")

        post_id = row['post_id']

        # LLM responses
        for key, short in model_key_to_short.items():
            col = f'{key}_response'
            text = row.get(col, '')
            if text and isinstance(text, str):
                metrics = compute_permission_metrics(text)
                if metrics.get('valid'):
                    metrics['post_id'] = post_id
                    metrics['source'] = short
                    del metrics['valid']
                    all_metrics.append(metrics)

        # Human response
        human_text = row.get('human_response', '')
        if human_text and isinstance(human_text, str):
            metrics = compute_permission_metrics(human_text)
            if metrics.get('valid'):
                metrics['post_id'] = post_id
                metrics['source'] = 'human'
                del metrics['valid']
                all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n  Total rows: {len(metrics_df)}")

    # Save raw metrics
    metrics_df.to_parquet(OUTPUT_DIR / "permission_metrics.parquet", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "permission_metrics.csv", index=False)
    print(f"  Saved to: {OUTPUT_DIR}")

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Summary statistics
    print("\n--- MEAN PERMISSION RATES (per 1000 chars) ---")
    summary = metrics_df.groupby('source').agg({
        'modal_permission_per_1k': ['mean', 'std'],
        'negated_obligation_per_1k': ['mean', 'std'],
        'entitlement_per_1k': ['mean', 'std'],
        'total_permission_per_1k': ['mean', 'std'],
    }).round(4)
    print(summary.to_string())

    # Effect sizes vs human
    print("\n--- EFFECT SIZES (Cohen's d, LLM vs Human) ---")
    human_data = metrics_df[metrics_df['source'] == 'human']

    effect_sizes = []
    for source in ['gemini', 'deepseek', 'ministral', 'gpt_nano']:
        llm_data = metrics_df[metrics_df['source'] == source]

        for metric in ['modal_permission_per_1k', 'negated_obligation_per_1k',
                       'entitlement_per_1k', 'total_permission_per_1k']:
            d = cohens_d(llm_data[metric], human_data[metric])
            _, p = stats.mannwhitneyu(llm_data[metric], human_data[metric])
            effect_sizes.append({
                'source': source,
                'metric': metric.replace('_per_1k', ''),
                'cohens_d': d,
                'p_value': p,
            })

    effects_df = pd.DataFrame(effect_sizes)
    print(effects_df.pivot(index='metric', columns='source', values='cohens_d').round(3).to_string())

    # Average across LLMs
    print("\n--- AVERAGE LLM vs HUMAN ---")
    llm_combined = metrics_df[metrics_df['source'] != 'human']

    for metric in ['modal_permission_per_1k', 'negated_obligation_per_1k',
                   'entitlement_per_1k', 'total_permission_per_1k']:
        human_mean = human_data[metric].mean()
        llm_mean = llm_combined[metric].mean()
        d = cohens_d(llm_combined[metric], human_data[metric])
        ratio = llm_mean / human_mean if human_mean > 0 else np.inf
        print(f"  {metric.replace('_per_1k', ''):25s}: Human={human_mean:.4f}, LLM={llm_mean:.4f}, ratio={ratio:.2f}x, d={d:.3f}")

    # Sample examples
    print("\n--- SAMPLE PERMISSION PHRASES FROM HUMAN ADVICE ---")
    human_texts = df['human_response'].dropna().head(500)
    all_matches = []
    for text in human_texts:
        for cat_name, patterns in [
            ('modal', MODAL_PERMISSION_PATTERNS),
            ('negated', NEGATED_OBLIGATION_PATTERNS),
            ('entitlement', ENTITLEMENT_PATTERNS),
        ]:
            matches = extract_pattern_matches(text, patterns)
            for m in matches:
                all_matches.append((cat_name, m))

    from collections import Counter
    for cat in ['modal', 'negated', 'entitlement']:
        cat_matches = [m for c, m in all_matches if c == cat]
        print(f"\n  {cat.upper()} (top 5):")
        for phrase, count in Counter(cat_matches).most_common(5):
            print(f"    '{phrase}': {count}")

    # Save summary
    summary_stats = {
        'human_total_per_1k': human_data['total_permission_per_1k'].mean(),
        'llm_avg_total_per_1k': llm_combined['total_permission_per_1k'].mean(),
        'effect_size_total': cohens_d(llm_combined['total_permission_per_1k'],
                                       human_data['total_permission_per_1k']),
    }
    for source in ['gemini', 'deepseek', 'ministral', 'gpt_nano']:
        llm_data = metrics_df[metrics_df['source'] == source]
        summary_stats[f'{source}_total_per_1k'] = llm_data['total_permission_per_1k'].mean()
        summary_stats[f'{source}_effect_d'] = cohens_d(llm_data['total_permission_per_1k'],
                                                        human_data['total_permission_per_1k'])

    with open(OUTPUT_DIR / "summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)

    effects_df.to_csv(OUTPUT_DIR / "effect_sizes.csv", index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
