#!/usr/bin/env python3
"""
Prepare a stratified sample of advice responses for manual validation.

Creates a CSV file with responses to be annotated for:
1. Leave/stay orientation (1-5 scale: 1=strongly stay, 5=strongly leave)
2. Certainty level (1-5 scale: 1=very hedged, 5=very certain)
3. Therapeutic framing (0=absent, 1=present)

Usage:
    python prepare_validation_sample.py [--n-posts 100]

Output: data/validation/validation_sample_<timestamp>.csv
"""

import json
import random
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
ADVICE_DIR = DATA_DIR / "llm_advice"
METRICS_DIR = DATA_DIR / "advice_metrics"
OUTPUT_DIR = DATA_DIR / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_advice_data():
    """Load advice responses from checkpoints."""
    model_names = {
        'google_gemini-2-5-flash-lite': 'gemini',
        'deepseek_deepseek-chat-v3-0324': 'deepseek',
        'mistralai_ministral-8b': 'ministral',
        'openai_gpt-4-1-nano': 'gpt_nano',
    }

    all_data = {}
    for f in sorted(ADVICE_DIR.glob('checkpoint_*.json')):
        model_key = f.stem.replace('checkpoint_', '')
        short_name = model_names.get(model_key, model_key)
        with open(f) as fh:
            d = json.load(fh)
        for post_id, result in d.get('completed', {}).items():
            if post_id not in all_data:
                all_data[post_id] = {'post_id': post_id}
            if 'error' not in result:
                all_data[post_id][f'{short_name}_response'] = result.get('response', '')

    return pd.DataFrame(list(all_data.values()))


def load_human_comments():
    """Load top human comment for each eligible post."""
    comments_df = pd.read_parquet(DATA_DIR / "r_relationship_advice_comments_cleaned.parquet")
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)

    quality = comments_df[
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= 5) &
        (comments_df['body'].str.len() >= 200)
    ]

    top_comments = quality.sort_values('score', ascending=False).groupby('post_id').first()
    top_comments = top_comments[['body', 'score']].reset_index()
    top_comments.columns = ['post_id', 'human_response', 'human_score']

    return top_comments


def load_topic_assignments():
    """Load topic assignments for stratification."""
    assign_dir = DATA_DIR / "topic_assignment"
    assign_files = list(assign_dir.glob("topic_assignments_*.parquet"))
    if not assign_files:
        return None

    df = pd.read_parquet(sorted(assign_files)[-1])
    # Use Gemini assignments as primary
    if 'gemini_primary' in df.columns:
        df['topic'] = df['gemini_primary']
    elif 'primary_topic' in df.columns:
        df['topic'] = df['primary_topic']
    else:
        return None

    return df[['post_id', 'topic']]


def load_metrics():
    """Load computed metrics for reference."""
    metrics_files = list(METRICS_DIR.glob("advice_metrics_*.parquet"))
    if not metrics_files:
        return None
    return pd.read_parquet(sorted(metrics_files)[-1])


def main():
    parser = argparse.ArgumentParser(description="Prepare validation sample")
    parser.add_argument("--n-posts", type=int, default=100,
                        help="Number of posts to sample (will have 2 responses each)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading data...")
    advice_df = load_advice_data()
    human_df = load_human_comments()
    topics_df = load_topic_assignments()
    metrics_df = load_metrics()

    # Merge
    df = advice_df.merge(human_df, on='post_id', how='inner')
    if topics_df is not None:
        df = df.merge(topics_df, on='post_id', how='left')
    else:
        df['topic'] = 'unknown'

    print(f"  Total posts with both LLM and human advice: {len(df)}")

    # Stratified sample by topic
    # Get top 20 topics by frequency for stratification
    topic_counts = df['topic'].value_counts()
    top_topics = topic_counts.head(20).index.tolist()

    # Sample proportionally from top topics, remainder from others
    sampled_ids = []
    n_per_topic = max(1, args.n_posts // 20)

    for topic in top_topics:
        topic_posts = df[df['topic'] == topic]['post_id'].tolist()
        n_sample = min(n_per_topic, len(topic_posts))
        sampled_ids.extend(random.sample(topic_posts, n_sample))

    # Fill remainder randomly
    remaining = args.n_posts - len(sampled_ids)
    if remaining > 0:
        available = df[~df['post_id'].isin(sampled_ids)]['post_id'].tolist()
        sampled_ids.extend(random.sample(available, min(remaining, len(available))))

    sampled_ids = sampled_ids[:args.n_posts]
    print(f"  Sampled {len(sampled_ids)} posts")

    # Create long-form annotation file
    # Each row is one response (either human or one LLM)
    annotation_rows = []

    for post_id in sampled_ids:
        row = df[df['post_id'] == post_id].iloc[0]
        topic = row.get('topic', 'unknown')

        # Human response
        if pd.notna(row.get('human_response')) and len(str(row['human_response'])) > 50:
            annotation_rows.append({
                'annotation_id': f"{post_id}_human",
                'post_id': post_id,
                'topic': topic,
                'source': 'human',
                'response_text': row['human_response'],
                'leave_stay_orientation': '',  # 1-5 scale
                'certainty_level': '',  # 1-5 scale
                'therapeutic_framing': '',  # 0 or 1
                'notes': ''
            })

        # One random LLM response (to keep annotation tractable)
        llm_sources = ['gemini', 'deepseek', 'ministral', 'gpt_nano']
        random.shuffle(llm_sources)

        for source in llm_sources:
            resp = row.get(f'{source}_response')
            if pd.notna(resp) and len(str(resp)) > 50:
                annotation_rows.append({
                    'annotation_id': f"{post_id}_{source}",
                    'post_id': post_id,
                    'topic': topic,
                    'source': source,
                    'response_text': str(resp),
                    'leave_stay_orientation': '',
                    'certainty_level': '',
                    'therapeutic_framing': '',
                    'notes': ''
                })
                break  # Only one LLM per post

    annotation_df = pd.DataFrame(annotation_rows)

    # Shuffle to avoid source clustering
    annotation_df = annotation_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"  Total responses to annotate: {len(annotation_df)}")
    print(f"  By source: {annotation_df['source'].value_counts().to_dict()}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f"validation_sample_{timestamp}.csv"
    annotation_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Also save the lexicon scores for these responses (for later correlation)
    if metrics_df is not None:
        # Get metrics for sampled posts
        sample_metrics = metrics_df[metrics_df['post_id'].isin(sampled_ids)].copy()
        sample_metrics['annotation_id'] = sample_metrics['post_id'] + '_' + sample_metrics['source']
        metrics_out = OUTPUT_DIR / f"validation_metrics_{timestamp}.csv"
        sample_metrics.to_csv(metrics_out, index=False)
        print(f"  Saved metrics: {metrics_out}")

    print("\n" + "=" * 60)
    print("ANNOTATION INSTRUCTIONS")
    print("=" * 60)
    print("""
For each response, code:

1. LEAVE_STAY_ORIENTATION (1-5):
   1 = Strongly stay-oriented ("work it out", "communicate", "give it time")
   2 = Somewhat stay-oriented
   3 = Neutral or balanced
   4 = Somewhat leave-oriented
   5 = Strongly leave-oriented ("leave", "break up", "run", "end it")

2. CERTAINTY_LEVEL (1-5):
   1 = Very hedged ("maybe", "perhaps", "it could be", lots of qualifiers)
   2 = Somewhat hedged
   3 = Neutral
   4 = Somewhat certain
   5 = Very certain ("definitely", "clearly", "you need to", direct statements)

3. THERAPEUTIC_FRAMING (0 or 1):
   0 = Absent (practical advice, direct recommendations, no therapy-speak)
   1 = Present (mentions boundaries, self-worth, healing, trauma, therapy, etc.)

Fill in the CSV and save as validation_sample_annotated.csv
""")


if __name__ == "__main__":
    main()
