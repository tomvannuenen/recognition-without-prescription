#!/usr/bin/env python3
"""
Generate LLM advice for eligible posts (11,565 posts with quality human comments).

Sends each post body to multiple LLMs and collects their natural responses.
Posts are sent WITHOUT the Reddit title to avoid activating genre-specific patterns.

Key design decisions (from methods.tex):
- Only post body text, no title (avoids "[24F] boyfriend [26M]" format cueing)
- No prompt scaffolding ("respond as a forum commenter", "what advice would you give")
- What the model *chooses* to do is itself data about its defaults

Usage:
    python generate_llm_advice.py                     # Run all models
    python generate_llm_advice.py --model openai/gpt-4.1-nano  # Run one model
    python generate_llm_advice.py --dry-run            # Show plan
    python generate_llm_advice.py --merge-only         # Merge existing checkpoints
"""

import os
import json
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Configuration
DEFAULT_MODELS = [
    "google/gemini-2.5-flash-lite",   # Google (US)
    "deepseek/deepseek-chat-v3-0324", # DeepSeek (China)
    "mistralai/ministral-8b",         # Mistral (Europe)
    "openai/gpt-4.1-nano",            # OpenAI (US)
]

MAX_CONCURRENT = 10       # Max simultaneous API calls
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120
CHECKPOINT_INTERVAL = 100
MAX_POST_CHARS = 8000     # Truncate long posts

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
POSTS_FILE = DATA_DIR / "r_relationship_advice_posts_cleaned.parquet"
COMMENTS_FILE = DATA_DIR / "r_relationship_advice_comments_cleaned.parquet"
OUTPUT_DIR = DATA_DIR / "llm_advice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_eligible_post_ids() -> set:
    """Get post IDs with quality comments (same filter as multi_model_assignment)."""
    comments_df = pd.read_parquet(COMMENTS_FILE)
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)

    quality = comments_df[
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= 5) &
        (comments_df['body'].str.len() >= 200)
    ]
    return set(quality['post_id'].unique())


def load_checkpoint(model_key: str) -> dict:
    """Load per-model checkpoint."""
    checkpoint_file = OUTPUT_DIR / f"checkpoint_{model_key}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed": {}}


def save_checkpoint(model_key: str, checkpoint: dict):
    """Save per-model checkpoint."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    checkpoint_file = OUTPUT_DIR / f"checkpoint_{model_key}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)


async def call_openrouter(
    session: aiohttp.ClientSession,
    model: str,
    post_id: str,
    post_body: str,
    api_key: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """Call OpenRouter API to generate advice for a single post."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/relationship-advice-research",
        "X-Title": "LLM Advice Generation",
    }

    # Truncate long posts
    if len(post_body) > MAX_POST_CHARS:
        post_body = post_body[:MAX_POST_CHARS] + "..."

    # Just the post body — no framing, no instructions
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": post_body}],
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
                ) as resp:
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
                        return {
                            "post_id": post_id,
                            "error": f"HTTP {resp.status}: {error_text[:200]}"
                        }
            except asyncio.TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    return {"post_id": post_id, "error": "timeout"}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {"post_id": post_id, "error": str(e)[:200]}
                await asyncio.sleep(1)

    return {"post_id": post_id, "error": "max_retries_exceeded"}


async def run_model(model_id: str, posts_df: pd.DataFrame, api_key: str,
                    workers: int = MAX_CONCURRENT):
    """Run advice generation for a single model on all remaining posts."""
    model_key = model_id.replace("/", "_").replace(".", "-")

    print(f"\n{'='*60}")
    print(f"MODEL: {model_id}")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = load_checkpoint(model_key)
    already_done = set(checkpoint.get("completed", {}).keys())
    remaining_df = posts_df[~posts_df['id'].isin(already_done)]

    print(f"  Total posts: {len(posts_df)}")
    print(f"  Already done: {len(already_done)}")
    print(f"  Remaining: {len(remaining_df)}")

    if len(remaining_df) == 0:
        print("  All posts already processed!")
        return checkpoint["completed"]

    semaphore = asyncio.Semaphore(workers)
    processed = 0
    errors = 0

    async with aiohttp.ClientSession() as session:
        # Process in batches for checkpointing
        batch_size = CHECKPOINT_INTERVAL
        rows = list(remaining_df.iterrows())

        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start:batch_start + batch_size]

            # Fire all tasks in this batch concurrently
            tasks = [
                call_openrouter(
                    session, model_id,
                    row['id'], row['selftext'],
                    api_key, semaphore
                )
                for _, row in batch
            ]
            results = await asyncio.gather(*tasks)

            # Store results
            for result in results:
                checkpoint["completed"][result["post_id"]] = result
                processed += 1
                if "error" in result:
                    errors += 1

            # Save checkpoint after each batch
            save_checkpoint(model_key, checkpoint)
            total_done = len(checkpoint["completed"])
            print(f"  Processed {total_done}/{len(posts_df)} "
                  f"(batch {batch_start//batch_size + 1}, {errors} errors total)")

    print(f"  Complete: {processed} processed, {errors} errors")
    return checkpoint["completed"]


def merge_results(models: list):
    """Merge all per-model checkpoints into final output."""
    print(f"\n{'='*60}")
    print("MERGING RESULTS")
    print(f"{'='*60}")

    all_data = {}  # post_id -> {model_key: response, ...}

    for model_id in models:
        model_key = model_id.replace("/", "_").replace(".", "-")
        checkpoint = load_checkpoint(model_key)
        completed = checkpoint.get("completed", {})

        if not completed:
            print(f"  {model_id}: no results")
            continue

        success = sum(1 for v in completed.values() if "error" not in v)
        errors = len(completed) - success
        print(f"  {model_id}: {success} responses, {errors} errors")

        for post_id, result in completed.items():
            if post_id not in all_data:
                all_data[post_id] = {"post_id": post_id}
            if "error" in result:
                all_data[post_id][f"{model_key}_response"] = None
                all_data[post_id][f"{model_key}_error"] = result["error"]
            else:
                all_data[post_id][f"{model_key}_response"] = result.get("response")
                all_data[post_id][f"{model_key}_error"] = None
                all_data[post_id][f"{model_key}_finish_reason"] = result.get("finish_reason")

    if not all_data:
        print("  No results to merge!")
        return

    # Build DataFrame
    results_df = pd.DataFrame(list(all_data.values()))

    # Merge with post metadata
    posts_df = pd.read_parquet(POSTS_FILE)
    posts_df = posts_df.rename(columns={"id": "post_id"})
    keep_cols = ["post_id", "title", "selftext", "author", "score",
                 "num_comments", "created_utc"]
    posts_meta = posts_df[[c for c in keep_cols if c in posts_df.columns]]
    merged = results_df.merge(posts_meta, on="post_id", how="left")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parquet_path = OUTPUT_DIR / f"llm_advice_{timestamp}.parquet"
    csv_path = OUTPUT_DIR / f"llm_advice_{timestamp}.csv"
    merged.to_parquet(parquet_path, index=False)
    merged.to_csv(csv_path, index=False)

    print(f"\n  Posts with responses: {len(merged)}")
    print(f"  Saved: {parquet_path}")
    print(f"  CSV: {csv_path}")

    # Per-model success rates and truncation check
    print("\n  Per-model stats:")
    for model_id in models:
        model_key = model_id.replace("/", "_").replace(".", "-")
        resp_col = f"{model_key}_response"
        fr_col = f"{model_key}_finish_reason"
        if resp_col in merged.columns:
            success = merged[resp_col].notna().sum()
            truncated = 0
            if fr_col in merged.columns:
                truncated = (merged[fr_col] == "length").sum()
            trunc_str = f", {truncated} truncated" if truncated > 0 else ""
            print(f"    {model_id}: {success}/{len(merged)} OK{trunc_str}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Generate LLM advice for eligible posts")
    parser.add_argument("--model", type=str, help="Run single model only")
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--limit", type=int, help="Process only N posts (for testing)")
    parser.add_argument("--merge-only", action="store_true", help="Just merge existing results")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    if args.merge_only:
        merge_results(DEFAULT_MODELS)
        return

    # Load eligible posts
    print("Loading eligible posts...")
    eligible_ids = get_eligible_post_ids()
    posts_df = pd.read_parquet(POSTS_FILE)
    posts_df = posts_df[posts_df['id'].isin(eligible_ids)]
    print(f"  Eligible posts: {len(posts_df)}")

    if args.limit:
        posts_df = posts_df.head(args.limit)
        print(f"  LIMITED TO: {len(posts_df)} posts (test mode)")

    # Determine which models to run
    models_to_run = [args.model] if args.model else DEFAULT_MODELS

    # Show plan
    print(f"\n{'='*60}")
    print("EXECUTION PLAN")
    print(f"{'='*60}")
    total_remaining = 0
    for model_id in models_to_run:
        model_key = model_id.replace("/", "_").replace(".", "-")
        checkpoint = load_checkpoint(model_key)
        already_done = len(checkpoint.get("completed", {}))
        remaining = len(posts_df) - already_done
        total_remaining += max(remaining, 0)
        print(f"  {model_id}: {already_done} done, {remaining} remaining")
    print(f"\n  Total API calls remaining: {total_remaining}")
    print(f"  Concurrency: {args.workers}")

    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return

    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Run each model
    for model_id in models_to_run:
        asyncio.run(run_model(model_id, posts_df, api_key, workers=args.workers))

    # Merge all results
    merge_results(DEFAULT_MODELS)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
