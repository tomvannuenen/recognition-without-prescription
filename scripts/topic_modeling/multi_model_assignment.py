#!/usr/bin/env python3
"""
Multi-Model Topic Assignment.

Runs topic assignment with multiple models to enable:
- Inter-model agreement analysis
- Majority-vote consensus assignment
- Model-specific topic bias testing

Budget-optimized design:
- 3 cheap models (GPT, Ministral, DeepSeek) on all eligible posts (11,565)
- Gemini 2.5 Flash Lite already assigned all 32,630 posts (baseline)
- Total: 4 models for consensus (one per provider)

Usage:
    python multi_model_assignment.py                    # Run all models
    python multi_model_assignment.py --model google/gemini-2.0-flash-001  # Run one model
    python multi_model_assignment.py --merge-only       # Just merge existing results
    python multi_model_assignment.py --dry-run          # Show plan without executing
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
TAXONOMY_FILE = DATA_DIR / "topic_consolidated" / "taxonomy_70_complete.json"
POSTS_FILE = DATA_DIR / "r_relationship_advice_posts_cleaned.parquet"
SAMPLE_DIR = DATA_DIR / "stratified_sample"
OUTPUT_DIR = DATA_DIR / "multi_model_assignment"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration: model_id -> scope
# "eligible" = all posts with quality comments (11,565)
# "sample" = stratified sample only (1,447)
MODEL_SCOPES = {
    "openai/gpt-4.1-nano": "eligible",             # $3.00
    "mistralai/ministral-8b": "eligible",           # $3.00
    "deepseek/deepseek-chat-v3-0324": "eligible",  # $5.80
}

ASSIGNMENT_PROMPT = """You are assigning topic labels to a relationship advice post.

Given the post below, select 1-3 topics from the taxonomy that best describe the ADVICE CONTEXT - what type of guidance this person needs.

Rules:
- Select the PRIMARY topic (most relevant) first
- Add 1-2 SECONDARY topics only if clearly applicable (not just tangentially related)
- Focus on the core issue, not minor details
- If only one topic applies, that's fine

TAXONOMY ({n_topics} topics):
{taxonomy}

POST:
{post}

Respond with JSON only:
{{
  "primary_topic": "most_relevant_topic_label",
  "secondary_topics": ["other_topic_1", "other_topic_2"],
  "reasoning": "Brief explanation of why these topics apply"
}}"""

MAX_WORKERS = 10
CHECKPOINT_INTERVAL = 100


def load_taxonomy() -> tuple:
    """Load taxonomy and format for prompt."""
    with open(TAXONOMY_FILE) as f:
        data = json.load(f)

    taxonomy = data.get("taxonomy", [])
    valid_labels = set()
    lines = []

    for cat in taxonomy:
        label = cat.get("topic", "")
        desc = cat.get("description", "")
        valid_labels.add(label)
        lines.append(f"- {label}: {desc}")

    return valid_labels, "\n".join(lines)


def get_eligible_post_ids() -> set:
    """Get post IDs that have quality comments (score >= 5, length >= 200)."""
    comments_df = pd.read_parquet(DATA_DIR / "r_relationship_advice_comments_cleaned.parquet")
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)

    quality = comments_df[
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= 5) &
        (comments_df['body'].str.len() >= 200)
    ]

    return set(quality['post_id'].unique())


def get_sample_post_ids() -> set:
    """Get post IDs from the stratified sample."""
    sample_files = list(SAMPLE_DIR.glob("stratified_sample_*.parquet"))
    if not sample_files:
        raise FileNotFoundError(f"No sample files in {SAMPLE_DIR}")
    latest = max(sample_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(latest)
    return set(df['post_id'].unique())


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


def extract_json(text: str) -> dict:
    """Extract JSON object from text that may contain markdown or extra content."""
    import re
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found", text, 0)


def assign_one_post(client, model, post_id, post_text,
                    taxonomy_formatted, valid_labels, n_topics) -> dict:
    """Assign topics to a single post."""
    if len(post_text) > 8000:
        post_text = post_text[:8000] + "..."

    prompt = ASSIGNMENT_PROMPT.format(
        n_topics=n_topics,
        taxonomy=taxonomy_formatted,
        post=post_text
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
            extra_headers={
                "HTTP-Referer": "https://github.com/relationship-advice-research",
                "X-Title": "Multi-Model Topic Assignment"
            }
        )

        content = response.choices[0].message.content
        if not content or not content.strip():
            return {"post_id": post_id, "error": "empty_response"}

        result = extract_json(content)
        primary = result.get("primary_topic", "")
        secondary = result.get("secondary_topics", [])
        reasoning = result.get("reasoning", "")

        # Validate primary
        if primary not in valid_labels:
            primary_norm = primary.lower().replace(" ", "_").replace("-", "_")
            matches = [l for l in valid_labels
                       if primary_norm in l.lower() or l.lower() in primary_norm]
            primary = matches[0] if matches else "unknown"

        valid_secondary = [t for t in secondary if t in valid_labels]

        return {
            "post_id": post_id,
            "primary_topic": primary,
            "secondary_topics": valid_secondary,
            "n_topics": 1 + len(valid_secondary),
            "reasoning": reasoning
        }

    except json.JSONDecodeError as e:
        return {"post_id": post_id, "error": f"json_parse: {e}"}
    except Exception as e:
        return {"post_id": post_id, "error": str(e)}


def run_model(model_id: str, post_ids: set, posts_df: pd.DataFrame,
              valid_labels: set, taxonomy_formatted: str,
              workers: int = MAX_WORKERS):
    """Run assignment for a single model on given posts."""
    model_key = model_id.replace("/", "_").replace(".", "-")

    print(f"\n{'='*60}")
    print(f"MODEL: {model_id}")
    print(f"Posts to assign: {len(post_ids)}")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = load_checkpoint(model_key)
    already_done = set(checkpoint.get("completed", {}).keys())
    remaining_ids = post_ids - already_done

    if already_done:
        print(f"  Resuming: {len(already_done)} already done, {len(remaining_ids)} remaining")

    if not remaining_ids:
        print("  All posts already processed!")
        return checkpoint["completed"]

    # Filter posts to remaining
    remaining_df = posts_df[posts_df['id'].isin(remaining_ids)]
    print(f"  Processing {len(remaining_df)} posts with {workers} workers...")

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    processed = 0
    errors = 0

    def process_post(row):
        return assign_one_post(
            client, model_id,
            row['id'], row['selftext'],
            taxonomy_formatted, valid_labels, len(valid_labels)
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_post, row): row['id']
            for _, row in remaining_df.iterrows()
        }

        for future in as_completed(futures):
            result = future.result()
            checkpoint["completed"][result["post_id"]] = result
            processed += 1

            if "error" in result:
                errors += 1

            if processed % 50 == 0:
                print(f"  Processed {processed}/{len(remaining_df)} ({errors} errors)")

            if processed % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(model_key, checkpoint)

    save_checkpoint(model_key, checkpoint)
    print(f"  Complete: {processed} processed, {errors} errors")

    return checkpoint["completed"]


def merge_results():
    """Merge all per-model checkpoints into a combined output file."""
    print(f"\n{'='*60}")
    print("MERGING RESULTS")
    print(f"{'='*60}")

    # Load existing Gemini 2.5 Flash Lite assignment as baseline
    gemini_fl_file = list((DATA_DIR / "topic_assignment").glob(
        "posts_with_topics_google_gemini-2.5-flash-lite_*.parquet"
    ))
    if gemini_fl_file:
        baseline = pd.read_parquet(max(gemini_fl_file, key=lambda p: p.stat().st_mtime))
        baseline_assignments = baseline[['post_id', 'primary_topic', 'secondary_topic_1', 'secondary_topic_2']].copy()
        baseline_assignments.columns = ['post_id', 'gemini-2-5-flash-lite_primary',
                                        'gemini-2-5-flash-lite_secondary_1',
                                        'gemini-2-5-flash-lite_secondary_2']
        print(f"  Baseline (Gemini 2.5 FL): {len(baseline_assignments)} posts")
    else:
        baseline_assignments = None
        print("  WARNING: No baseline Gemini 2.5 FL assignment found")

    # Load each model's checkpoint
    all_assignments = {}
    for model_id in MODEL_SCOPES:
        model_key = model_id.replace("/", "_").replace(".", "-")
        checkpoint = load_checkpoint(model_key)
        completed = checkpoint.get("completed", {})

        if not completed:
            print(f"  {model_id}: no results found")
            continue

        # Convert to DataFrame
        rows = []
        for post_id, result in completed.items():
            if "error" in result:
                rows.append({"post_id": post_id, "primary_topic": "", "error": result["error"]})
            else:
                sec = result.get("secondary_topics", [])
                rows.append({
                    "post_id": post_id,
                    "primary_topic": result.get("primary_topic", ""),
                    "secondary_1": sec[0] if len(sec) > 0 else "",
                    "secondary_2": sec[1] if len(sec) > 1 else "",
                })

        model_df = pd.DataFrame(rows)
        col_prefix = model_key.replace("_", "-")
        model_df.columns = ['post_id'] + [f"{col_prefix}_{c}" for c in model_df.columns[1:]]
        all_assignments[model_id] = model_df
        print(f"  {model_id}: {len(model_df)} assignments")

    if not all_assignments:
        print("  No model results to merge!")
        return

    # Merge all on post_id
    merged = None
    for model_id, model_df in all_assignments.items():
        if merged is None:
            merged = model_df
        else:
            merged = merged.merge(model_df, on='post_id', how='outer')

    # Add baseline
    if baseline_assignments is not None:
        merged = merged.merge(baseline_assignments, on='post_id', how='outer')

    # Compute agreement metrics
    primary_cols = [c for c in merged.columns if c.endswith('_primary_topic') or c.endswith('_primary')]
    if primary_cols:
        def compute_agreement(row):
            topics = [row[c] for c in primary_cols if pd.notna(row.get(c)) and row.get(c) != '']
            if len(topics) < 2:
                return None
            # Majority vote
            from collections import Counter
            counts = Counter(topics)
            most_common = counts.most_common(1)[0]
            return {
                'majority_topic': most_common[0],
                'majority_count': most_common[1],
                'n_models': len(topics),
                'agreement_ratio': most_common[1] / len(topics)
            }

        print("\n  Computing agreement metrics...")
        agreement = merged.apply(compute_agreement, axis=1)
        agreement_df = pd.DataFrame(agreement.dropna().tolist(), index=agreement.dropna().index)
        for col in agreement_df.columns:
            merged[col] = agreement_df[col]

        # Summary
        if 'agreement_ratio' in merged.columns:
            avg_agreement = merged['agreement_ratio'].mean()
            full_agreement = (merged['agreement_ratio'] == 1.0).mean()
            print(f"  Average agreement ratio: {avg_agreement:.3f}")
            print(f"  Full agreement (all models same): {full_agreement:.1%}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = OUTPUT_DIR / f"multi_model_assignments_{timestamp}.parquet"
    merged.to_parquet(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    csv_path = OUTPUT_DIR / f"multi_model_assignments_{timestamp}.csv"
    merged.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Multi-model topic assignment")
    parser.add_argument("--model", type=str, help="Run single model only")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--merge-only", action="store_true", help="Just merge existing results")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    if args.merge_only:
        merge_results()
        return

    # Load taxonomy
    print("Loading taxonomy...")
    valid_labels, taxonomy_formatted = load_taxonomy()
    print(f"  {len(valid_labels)} topics")

    # Determine post scopes
    print("\nDetermining post scopes...")
    eligible_ids = get_eligible_post_ids()
    sample_ids = get_sample_post_ids()
    print(f"  Eligible posts (quality comments): {len(eligible_ids)}")
    print(f"  Sampled posts: {len(sample_ids)}")

    # Load all posts
    posts_df = pd.read_parquet(POSTS_FILE)
    # Filter to eligible
    posts_df = posts_df[posts_df['id'].isin(eligible_ids | sample_ids)]
    print(f"  Posts loaded: {len(posts_df)}")

    # Determine which models to run
    models_to_run = {args.model: MODEL_SCOPES.get(args.model, "sample")} if args.model else MODEL_SCOPES

    # Show plan
    print(f"\n{'='*60}")
    print("EXECUTION PLAN")
    print(f"{'='*60}")
    total_calls = 0
    for model_id, scope in models_to_run.items():
        n_posts = len(eligible_ids) if scope == "eligible" else len(sample_ids)
        model_key = model_id.replace("/", "_").replace(".", "-")
        checkpoint = load_checkpoint(model_key)
        already_done = len(checkpoint.get("completed", {}))
        remaining = n_posts - already_done
        total_calls += remaining
        print(f"  {model_id}: {scope} ({n_posts} posts, {already_done} done, {remaining} remaining)")
    print(f"\n  Total API calls: {total_calls}")

    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return

    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY not set")

    # Run each model
    for model_id, scope in models_to_run.items():
        post_ids = eligible_ids if scope == "eligible" else sample_ids
        run_model(model_id, post_ids, posts_df, valid_labels, taxonomy_formatted, args.workers)

    # Merge all results
    merge_results()

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
