#!/usr/bin/env python3
"""
Stratified sampling of posts for LLM advice generation.

Creates a balanced sample across all 70 topics with:
- Minimum 10 posts per topic (floor)
- Maximum 50 posts per topic (ceiling)
- Quality filter: posts must have a top comment with score >= 5 and length >= 200

Output includes both the post and its top human comment for comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Configuration
MIN_PER_TOPIC = 10
MAX_PER_TOPIC = 50
MIN_COMMENT_SCORE = 5
MIN_COMMENT_LENGTH = 200
RANDOM_SEED = 42

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
POSTS_FILE = DATA_DIR / 'topic_assignment' / 'posts_with_topics_google_gemini-2.5-flash-lite_20260121_094411.parquet'
COMMENTS_FILE = DATA_DIR / 'r_relationship_advice_comments_cleaned.parquet'
OUTPUT_DIR = DATA_DIR / 'stratified_sample'


def load_data():
    """Load posts and comments dataframes."""
    print("Loading data...")
    posts_df = pd.read_parquet(POSTS_FILE)
    comments_df = pd.read_parquet(COMMENTS_FILE)

    print(f"  Posts: {len(posts_df):,}")
    print(f"  Comments: {len(comments_df):,}")

    return posts_df, comments_df


def get_quality_top_comments(comments_df):
    """
    Filter to quality top-level comments and get the highest-scored per post.

    Quality criteria:
    - Top-level comment (direct reply to post)
    - Not from OP
    - Score >= MIN_COMMENT_SCORE
    - Length >= MIN_COMMENT_LENGTH
    """
    print(f"\nFiltering quality comments...")
    print(f"  Criteria: top-level, non-OP, score >= {MIN_COMMENT_SCORE}, length >= {MIN_COMMENT_LENGTH}")

    # Extract post_id from link_id (remove 't3_' prefix)
    comments_df = comments_df.copy()
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)

    # Filter to quality comments
    quality_comments = comments_df[
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= MIN_COMMENT_SCORE) &
        (comments_df['body'].str.len() >= MIN_COMMENT_LENGTH)
    ].copy()

    print(f"  Quality comments: {len(quality_comments):,}")

    # Get top comment per post (highest score)
    top_comments = (
        quality_comments
        .sort_values('score', ascending=False)
        .groupby('post_id')
        .first()
        .reset_index()
    )

    print(f"  Posts with quality top comment: {len(top_comments):,}")

    return top_comments[['post_id', 'id', 'author', 'body', 'score', 'created_utc']].rename(columns={
        'id': 'top_comment_id',
        'author': 'top_comment_author',
        'body': 'top_comment_body',
        'score': 'top_comment_score',
        'created_utc': 'top_comment_created_utc'
    })


def stratified_sample(posts_df, min_per_topic=MIN_PER_TOPIC, max_per_topic=MAX_PER_TOPIC, seed=RANDOM_SEED):
    """
    Create stratified sample with floor/ceiling per topic.

    Strategy:
    - Calculate proportional target based on topic frequency
    - Apply minimum (floor) and maximum (ceiling) constraints
    - Sample randomly within each topic
    """
    print(f"\nStratified sampling (min={min_per_topic}, max={max_per_topic})...")

    np.random.seed(seed)

    topic_counts = posts_df['primary_topic'].value_counts()
    total_posts = len(posts_df)

    samples = []
    stats = []

    for topic in sorted(topic_counts.index):
        available = topic_counts[topic]

        # Proportional target (scaled to ~2000 total)
        prop_target = int(available / total_posts * 2000)

        # Apply floor and ceiling
        target = max(min_per_topic, min(max_per_topic, prop_target))

        # Can't sample more than available
        actual_n = min(target, available)

        # Sample
        topic_posts = posts_df[posts_df['primary_topic'] == topic]
        if actual_n < len(topic_posts):
            sampled = topic_posts.sample(n=actual_n, random_state=seed)
        else:
            sampled = topic_posts

        samples.append(sampled)

        stats.append({
            'topic': topic,
            'available': available,
            'target': target,
            'sampled': len(sampled),
            'shortfall': target - len(sampled) if len(sampled) < target else 0
        })

    sample_df = pd.concat(samples, ignore_index=True)
    stats_df = pd.DataFrame(stats)

    print(f"  Total sampled: {len(sample_df):,} posts across {sample_df['primary_topic'].nunique()} topics")

    shortfall_topics = stats_df[stats_df['shortfall'] > 0]
    if len(shortfall_topics) > 0:
        print(f"  Topics with shortfall ({len(shortfall_topics)}):")
        for _, row in shortfall_topics.iterrows():
            print(f"    - {row['topic']}: {row['sampled']}/{row['target']}")

    return sample_df, stats_df


def main():
    """Main sampling pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    posts_df, comments_df = load_data()

    # Filter out error assignments
    posts_df = posts_df[~posts_df['primary_topic'].isin(['', 'unknown'])].copy()
    print(f"Posts after removing errors: {len(posts_df):,}")

    # Get quality top comments
    top_comments = get_quality_top_comments(comments_df)

    # Filter posts to those with quality comments
    posts_with_comments = posts_df[posts_df['post_id'].isin(top_comments['post_id'])].copy()
    print(f"Posts eligible for sampling: {len(posts_with_comments):,}")

    # Stratified sample
    sample_df, stats_df = stratified_sample(posts_with_comments)

    # Merge with top comments
    sample_with_comments = sample_df.merge(top_comments, on='post_id', how='left')

    # Select relevant columns for output
    output_columns = [
        'post_id', 'title', 'selftext', 'author', 'score', 'num_comments', 'created_utc',
        'primary_topic', 'secondary_topic_1', 'secondary_topic_2', 'n_topics',
        'top_comment_id', 'top_comment_author', 'top_comment_body', 'top_comment_score'
    ]

    # Filter to columns that exist
    output_columns = [c for c in output_columns if c in sample_with_comments.columns]
    output_df = sample_with_comments[output_columns].copy()

    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save outputs
    parquet_path = OUTPUT_DIR / f'stratified_sample_{timestamp}.parquet'
    csv_path = OUTPUT_DIR / f'stratified_sample_{timestamp}.csv'
    stats_path = OUTPUT_DIR / f'sampling_stats_{timestamp}.csv'

    output_df.to_parquet(parquet_path, index=False)
    output_df.to_csv(csv_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"  Sample: {parquet_path}")
    print(f"  Sample (CSV): {csv_path}")
    print(f"  Stats: {stats_path}")

    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Total posts sampled: {len(output_df):,}")
    print(f"  Topics covered: {output_df['primary_topic'].nunique()}")
    print(f"  Topics with shortfall: {(stats_df['shortfall'] > 0).sum()}")
    print(f"\n  Sample by topic (top 10):")
    for topic, count in output_df['primary_topic'].value_counts().head(10).items():
        print(f"    {topic}: {count}")

    # Save metadata
    metadata = {
        'created': timestamp,
        'config': {
            'min_per_topic': MIN_PER_TOPIC,
            'max_per_topic': MAX_PER_TOPIC,
            'min_comment_score': MIN_COMMENT_SCORE,
            'min_comment_length': MIN_COMMENT_LENGTH,
            'random_seed': RANDOM_SEED
        },
        'stats': {
            'total_posts_sampled': len(output_df),
            'topics_covered': int(output_df['primary_topic'].nunique()),
            'topics_with_shortfall': int((stats_df['shortfall'] > 0).sum()),
            'posts_eligible': len(posts_with_comments)
        }
    }

    metadata_path = OUTPUT_DIR / f'sampling_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Metadata: {metadata_path}")

    return output_df, stats_df


if __name__ == '__main__':
    main()
