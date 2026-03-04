#!/usr/bin/env python3
"""
Persona Prompting Sensitivity Check for LLM Advice Generation

Tests whether prompting LLMs with "Respond as a commenter on r/relationship_advice
would." narrows the divergence from community norms.

Theoretical motivation:
- If persona prompting substantially narrows the gap, it suggests the model HAS
  knowledge of community norms but doesn't deploy it by default (alignment choice)
- If the gap persists, it suggests either the model doesn't know the vernacular,
  or safety alignment actively suppresses it even when prompted

Design choice: We use a minimal prompt that doesn't explicitly instruct directiveness.
This tests whether the model can access community-specific vernacular competence
when pointed toward it, rather than just following instructions to be more directive.

Usage:
    python persona_prompting_check.py              # Generate with persona prompt
    python persona_prompting_check.py --compare    # Compare to default results
    python persona_prompting_check.py --dry-run    # Show plan only

Output: data/persona_prompting/
"""

import os
import re
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Configuration
SAMPLE_SIZE = 500
PERSONA_PROMPT = "Respond as a commenter on r/relationship_advice would."
MAX_CONCURRENT = 10
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120
CHECKPOINT_INTERVAL = 100

# Same models as main study
MODELS = [
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-chat-v3-0324",
    "mistralai/ministral-8b",
    "openai/gpt-4.1-nano",
]

MODEL_KEY_TO_SHORT = {
    'google_gemini-2-5-flash-lite': 'gemini',
    'deepseek_deepseek-chat-v3-0324': 'deepseek',
    'mistralai_ministral-8b': 'ministral',
    'openai_gpt-4-1-nano': 'gpt_nano',
}

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "persona_prompting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSTS_FILE = DATA_DIR / "r_relationship_advice_posts_cleaned.parquet"
COMMENTS_FILE = DATA_DIR / "r_relationship_advice_comments_cleaned.parquet"
ORIGINAL_ADVICE_DIR = DATA_DIR / "llm_advice"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# LEXICONS (same as compute_advice_metrics.py)
# ============================================================================

HEDGE_WORDS = {
    'might', 'could', 'may', 'perhaps', 'possibly', 'maybe', 'probably',
    'seemingly', 'apparently', 'arguably', 'potentially', 'somewhat',
}
BOOSTER_WORDS = {
    'clearly', 'definitely', 'obviously', 'absolutely', 'certainly',
    'undoubtedly', 'always', 'never', 'completely', 'totally',
}

LEAVE_WORDS = [
    'leave', 'break up', 'breakup', 'divorce', 'end it', 'walk away',
    'dump', 'move on', 'get out', 'run', 'separate', 'end the relationship',
]
STAY_WORDS = [
    'stay', 'work on', 'communicate', 'work it out', 'couples therapy',
    'marriage counseling', 'work through', 'reconcile', 'forgive',
]

ACTION_PERMISSION_PATTERNS = [
    r"\byou can leave\b", r"\byou can go\b", r"\byou can walk away\b",
    r"\byou can say no\b", r"\byou can end\b", r"\byou can break up\b",
    r"\byou don't have to\b", r"\byou don't need to\b", r"\byou don't owe\b",
    r"\byou're not obligated\b", r"\byou are not obligated\b",
]

ABSTRACT_ENTITLEMENT_PATTERNS = [
    r"\byou deserve\b", r"\byou have the right\b", r"\byou're worthy\b",
    r"\byou are worthy\b", r"\byou matter\b", r"\byou're worth\b",
]

# Therapy language (from compute_advice_metrics.py)
THERAPY_WORDS = [
    'boundaries', 'boundary', 'self-worth', 'self worth', 'self-care',
    'self care', 'healing', 'trauma', 'toxic', 'gaslighting', 'gaslight',
    'narcissist', 'narcissistic', 'red flag', 'red flags', 'dealbreaker',
    'deal breaker', 'therapy', 'therapist', 'counseling', 'counselor',
]


def count_patterns(text: str, patterns: list) -> int:
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def compute_key_metrics(text: str) -> dict:
    """Compute the key metrics we report in the paper."""
    if not text or len(text.strip()) < 50:
        return None

    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    # Hedging/certainty
    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    booster_count = sum(1 for w in words if w in BOOSTER_WORDS)
    total_stance = hedge_count + booster_count
    certainty_ratio = booster_count / total_stance if total_stance > 0 else 0.5

    # Leave/stay
    leave_count = count_patterns(text, LEAVE_WORDS)
    stay_count = count_patterns(text, STAY_WORDS)
    total_ls = leave_count + stay_count
    leave_ratio = leave_count / total_ls if total_ls > 0 else 0.5

    # Permission (action vs abstract)
    action_count = count_patterns(text, ACTION_PERMISSION_PATTERNS)
    abstract_count = count_patterns(text, ABSTRACT_ENTITLEMENT_PATTERNS)
    total_perm = action_count + abstract_count
    permission_ratio = action_count / total_perm if total_perm > 0 else 0.5

    # Therapy density
    therapy_count = count_patterns(text, THERAPY_WORDS)
    therapy_density = (therapy_count / word_count * 1000) if word_count > 0 else 0

    return {
        'certainty_ratio': certainty_ratio,
        'leave_ratio': leave_ratio,
        'permission_ratio': permission_ratio,
        'therapy_density': therapy_density,
        'hedge_count': hedge_count,
        'booster_count': booster_count,
        'leave_count': leave_count,
        'action_permission_count': action_count,
        'abstract_entitlement_count': abstract_count,
        'word_count': word_count,
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def get_eligible_post_ids() -> set:
    """Get post IDs with quality comments."""
    comments_df = pd.read_parquet(COMMENTS_FILE)
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)
    quality = comments_df[
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= 5) &
        (comments_df['body'].str.len() >= 200)
    ]
    return set(quality['post_id'].unique())


def load_sample(seed: int = 42) -> pd.DataFrame:
    """Load a random sample of eligible posts."""
    eligible_ids = get_eligible_post_ids()
    posts_df = pd.read_parquet(POSTS_FILE)
    posts_df = posts_df[posts_df['id'].isin(eligible_ids)]
    sample_df = posts_df.sample(n=min(SAMPLE_SIZE, len(posts_df)), random_state=seed)
    return sample_df


def load_human_comments(post_ids: set) -> dict:
    """Load top human comment for specified posts."""
    comments_df = pd.read_parquet(COMMENTS_FILE)
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)

    quality = comments_df[
        (comments_df['post_id'].isin(post_ids)) &
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= 5) &
        (comments_df['body'].str.len() >= 200)
    ]

    top_comments = quality.sort_values('score', ascending=False).groupby('post_id').first()
    return top_comments['body'].to_dict()


# ============================================================================
# API CALLS
# ============================================================================

async def call_openrouter_with_persona(session, model, post_id, post_body, api_key, semaphore):
    """Call OpenRouter API with persona system prompt."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/relationship-advice-research",
        "X-Title": "LLM Advice Persona Check",
    }

    if len(post_body) > 8000:
        post_body = post_body[:8000] + "..."

    # Key difference: system prompt with persona instruction
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PERSONA_PROMPT},
            {"role": "user", "content": post_body}
        ],
        "max_tokens": 4096,
        "temperature": 0.7,  # Same as original
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, headers=headers, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choice = data['choices'][0]
                        content = choice['message']['content']
                        usage = data.get('usage', {})
                        finish_reason = choice.get('finish_reason', 'unknown')
                        return {
                            "post_id": post_id,
                            "response": content,
                            "finish_reason": finish_reason,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                        }
                    elif resp.status == 429:
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"Rate limited on {model}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await resp.text()
                        return {"post_id": post_id, "error": f"HTTP {resp.status}: {error_text[:200]}"}
            except asyncio.TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    return {"post_id": post_id, "error": "timeout"}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {"post_id": post_id, "error": str(e)[:200]}
                await asyncio.sleep(1)

    return {"post_id": post_id, "error": "max_retries_exceeded"}


def load_checkpoint(model_key: str) -> dict:
    checkpoint_file = OUTPUT_DIR / f"checkpoint_persona_{model_key}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed": {}}


def save_checkpoint(model_key: str, checkpoint: dict):
    checkpoint["last_updated"] = datetime.now().isoformat()
    checkpoint_file = OUTPUT_DIR / f"checkpoint_persona_{model_key}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)


async def run_model(model_id: str, posts_df: pd.DataFrame, api_key: str, workers: int = MAX_CONCURRENT):
    """Generate advice for one model with persona prompt."""
    model_key = model_id.replace("/", "_").replace(".", "-")

    checkpoint = load_checkpoint(model_key)
    already_done = set(checkpoint.get("completed", {}).keys())
    remaining_df = posts_df[~posts_df['id'].isin(already_done)]

    print(f"\n{model_id}: {len(already_done)} done, {len(remaining_df)} remaining")

    if len(remaining_df) == 0:
        return checkpoint["completed"]

    semaphore = asyncio.Semaphore(workers)

    async with aiohttp.ClientSession() as session:
        rows = list(remaining_df.iterrows())

        for batch_start in range(0, len(rows), CHECKPOINT_INTERVAL):
            batch = rows[batch_start:batch_start + CHECKPOINT_INTERVAL]

            tasks = [
                call_openrouter_with_persona(
                    session, model_id, row['id'], row['selftext'], api_key, semaphore
                )
                for _, row in batch
            ]
            results = await asyncio.gather(*tasks)

            for result in results:
                checkpoint["completed"][result["post_id"]] = result

            save_checkpoint(model_key, checkpoint)
            print(f"  Batch {batch_start // CHECKPOINT_INTERVAL + 1}: {len(checkpoint['completed'])} total")

    return checkpoint["completed"]


# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def load_original_advice(post_ids: set) -> dict:
    """Load original (no persona) advice for comparison posts."""
    results = {short: {} for short in MODEL_KEY_TO_SHORT.values()}

    for f in ORIGINAL_ADVICE_DIR.glob('checkpoint_*.json'):
        model_key = f.stem.replace('checkpoint_', '')
        short_name = MODEL_KEY_TO_SHORT.get(model_key)

        if not short_name:
            continue

        with open(f) as fh:
            data = json.load(fh)

        for post_id, result in data.get('completed', {}).items():
            if post_id in post_ids and 'error' not in result:
                results[short_name][post_id] = result.get('response', '')

    return results


def run_comparison():
    """Compare persona-prompted results to default results."""
    print("\n" + "=" * 70)
    print("PERSONA PROMPTING SENSITIVITY CHECK")
    print(f"Prompt: \"{PERSONA_PROMPT}\"")
    print("=" * 70)

    # Load persona results
    persona_results = {}
    for f in OUTPUT_DIR.glob('checkpoint_persona_*.json'):
        model_key = f.stem.replace('checkpoint_persona_', '')
        short_name = MODEL_KEY_TO_SHORT.get(model_key)

        if not short_name:
            continue

        with open(f) as fh:
            data = json.load(fh)

        persona_results[short_name] = {
            pid: r.get('response', '')
            for pid, r in data.get('completed', {}).items()
            if 'error' not in r
        }

    if not persona_results:
        print("No persona results found. Run without --compare first.")
        return

    # Get post IDs
    all_post_ids = set()
    for model_results in persona_results.values():
        all_post_ids.update(model_results.keys())

    print(f"\nPosts with persona results: {len(all_post_ids)}")

    # Load default results for same posts
    default_results = load_original_advice(all_post_ids)

    # Load human comments
    human_comments = load_human_comments(all_post_ids)
    print(f"Human comments loaded: {len(human_comments)}")

    # Compute metrics for each condition
    metrics_persona = []
    metrics_default = []
    metrics_human = []

    for post_id in all_post_ids:
        # Human
        if post_id in human_comments:
            m = compute_key_metrics(human_comments[post_id])
            if m:
                m['post_id'] = post_id
                m['source'] = 'human'
                metrics_human.append(m)

        # Each model
        for model in MODEL_KEY_TO_SHORT.values():
            # Persona
            if model in persona_results and post_id in persona_results[model]:
                text = persona_results[model][post_id]
                m = compute_key_metrics(text)
                if m:
                    m['post_id'] = post_id
                    m['source'] = model
                    metrics_persona.append(m)

            # Default
            if model in default_results and post_id in default_results[model]:
                text = default_results[model][post_id]
                m = compute_key_metrics(text)
                if m:
                    m['post_id'] = post_id
                    m['source'] = model
                    metrics_default.append(m)

    df_persona = pd.DataFrame(metrics_persona)
    df_default = pd.DataFrame(metrics_default)
    df_human = pd.DataFrame(metrics_human)

    print(f"\nMetric rows: persona: {len(df_persona)}, default: {len(df_default)}, human: {len(df_human)}")

    # Compare key metrics
    key_metrics = ['certainty_ratio', 'leave_ratio', 'permission_ratio', 'therapy_density']

    print("\n" + "-" * 70)
    print("KEY METRIC COMPARISON")
    print("-" * 70)

    print("\n--- MEANS BY CONDITION ---")
    print(f"{'Metric':<20} {'Human':<10} {'Default':<12} {'Persona':<12} {'Gap Closed':<12}")
    print("-" * 66)

    comparison_results = []

    for metric in key_metrics:
        human_mean = df_human[metric].mean()
        default_mean = df_default[metric].mean()
        persona_mean = df_persona[metric].mean()

        # How much of the human-default gap did persona close?
        original_gap = abs(human_mean - default_mean)
        new_gap = abs(human_mean - persona_mean)
        gap_closed_pct = ((original_gap - new_gap) / original_gap * 100) if original_gap > 0 else 0

        print(f"{metric:<20} {human_mean:<10.3f} {default_mean:<12.3f} {persona_mean:<12.3f} {gap_closed_pct:<10.1f}%")

        comparison_results.append({
            'metric': metric,
            'human_mean': human_mean,
            'default_mean': default_mean,
            'persona_mean': persona_mean,
            'original_gap': original_gap,
            'new_gap': new_gap,
            'gap_closed_pct': gap_closed_pct,
        })

    print("\n--- PER-MODEL LEAVE RATIO ---")
    print(f"{'Model':<15} {'Default':<12} {'Persona':<12} {'Human':<12} {'Gap Closed':<12}")
    print("-" * 63)

    human_leave = df_human['leave_ratio'].mean()

    for model in MODEL_KEY_TO_SHORT.values():
        default_model = df_default[df_default['source'] == model]['leave_ratio'].mean()
        persona_model = df_persona[df_persona['source'] == model]['leave_ratio'].mean()

        original_gap = abs(human_leave - default_model)
        new_gap = abs(human_leave - persona_model)
        gap_closed = ((original_gap - new_gap) / original_gap * 100) if original_gap > 0 else 0

        print(f"{model:<15} {default_model:<12.3f} {persona_model:<12.3f} {human_leave:<12.3f} {gap_closed:<10.1f}%")

    # Save results
    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv(OUTPUT_DIR / "persona_comparison.csv", index=False)

    # Summary
    avg_gap_closed = results_df['gap_closed_pct'].mean()

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"\nPersona prompt: \"{PERSONA_PROMPT}\"")
    print(f"Average gap closed across metrics: {avg_gap_closed:.1f}%")

    if avg_gap_closed > 50:
        print("\nInterpretation: Substantial narrowing suggests the model HAS knowledge")
        print("of community norms but doesn't deploy it by default. The divergence")
        print("is a product of alignment choices, not architectural constraint.")
    elif avg_gap_closed > 20:
        print("\nInterpretation: Moderate narrowing suggests partial vernacular knowledge.")
        print("Some gap closure, but alignment still suppresses full community-style advice.")
    else:
        print("\nInterpretation: Minimal narrowing suggests either (a) the model lacks")
        print("community-specific vernacular knowledge, or (b) safety alignment actively")
        print("suppresses directive advice even when explicitly prompted.")

    print(f"\nResults saved to: {OUTPUT_DIR / 'persona_comparison.csv'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Persona prompting sensitivity check")
    parser.add_argument("--compare", action="store_true", help="Compare persona to default results")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT, help="Max concurrent API calls")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
        return

    # Load sample
    print(f"Loading random sample of {SAMPLE_SIZE} posts (seed={args.seed})...")
    sample_df = load_sample(seed=args.seed)
    print(f"  Sample size: {len(sample_df)}")

    # Save sample IDs for reproducibility
    sample_df[['id']].to_csv(OUTPUT_DIR / "sample_post_ids.csv", index=False)

    # Show plan
    print(f"\n{'='*60}")
    print("EXECUTION PLAN")
    print(f"{'='*60}")
    print(f"  Persona prompt: \"{PERSONA_PROMPT}\"")
    print(f"  Sample size: {len(sample_df)}")
    print(f"  Models: {len(MODELS)}")
    print(f"  Total API calls: {len(sample_df) * len(MODELS)}")

    for model_id in MODELS:
        model_key = model_id.replace("/", "_").replace(".", "-")
        checkpoint = load_checkpoint(model_key)
        already_done = len(checkpoint.get("completed", {}))
        remaining = len(sample_df) - already_done
        print(f"    {model_id}: {already_done} done, {remaining} remaining")

    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return

    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Run each model
    for model_id in MODELS:
        asyncio.run(run_model(model_id, sample_df, api_key, workers=args.workers))

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nRun with --compare to see comparison analysis.")


if __name__ == "__main__":
    main()
