#!/usr/bin/env python3
"""
Data cleaning script for r/relationship_advice posts and comments.

Filters out:
- Posts: empty, deleted, removed, or too short (< 1000 characters)
- Comments: deleted, removed, bot comments (AutoModerator), and bot commands

Also aligns comments to only include those from posts that survive cleaning.
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
POSTS_INPUT = DATA_DIR / "r_relationship_advice_posts-oct-dec-2025.parquet"
COMMENTS_INPUT = DATA_DIR / "r_relationship_advice_comments.parquet"
POSTS_OUTPUT = DATA_DIR / "r_relationship_advice_posts_cleaned.parquet"
COMMENTS_OUTPUT = DATA_DIR / "r_relationship_advice_comments_cleaned.parquet"

# Filtering constants
MIN_POST_LENGTH = 1000
MIN_COMMENT_LENGTH = 100
REMOVED_MARKERS = ["[removed]", "[deleted]"]
BOT_AUTHORS = ["AutoModerator", "bot-sleuth-bot"]
BOT_COMMANDS = [
    "updateme", "updateme!", "update me", "update me!",
    "!updateme", "remindme", "remindme!", "!remindme"
]


def clean_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean posts by removing:
    - Empty selftext
    - Deleted posts ([deleted])
    - Removed posts ([removed])
    - Posts with selftext < MIN_POST_LENGTH characters
    """
    original_count = len(df)

    # Remove null/empty selftext
    df = df[df["selftext"].notna() & (df["selftext"].str.strip() != "")]
    after_empty = len(df)

    # Remove deleted/removed posts
    df = df[~df["selftext"].isin(REMOVED_MARKERS)]
    after_removed = len(df)

    # Remove posts that are too short
    df = df[df["selftext"].str.len() >= MIN_POST_LENGTH]
    after_length = len(df)

    print(f"Posts cleaning summary:")
    print(f"  Original:              {original_count:>8,}")
    print(f"  After removing empty:  {after_empty:>8,} (-{original_count - after_empty:,})")
    print(f"  After removing del/rm: {after_removed:>8,} (-{after_empty - after_removed:,})")
    print(f"  After length filter:   {after_length:>8,} (-{after_removed - after_length:,})")
    print(f"  Total removed:         {original_count - after_length:>8,} ({(original_count - after_length) / original_count * 100:.1f}%)")
    print()

    return df


def clean_comments(df: pd.DataFrame, valid_post_ids: set) -> pd.DataFrame:
    """
    Clean comments by removing:
    - Empty body
    - Deleted comments ([deleted])
    - Removed comments ([removed])
    - Bot comments (AutoModerator, bot-sleuth-bot)
    - Bot commands (UpdateMe, RemindMe, etc.)
    - Short comments (< MIN_COMMENT_LENGTH characters)
    - Comments from posts that didn't survive cleaning
    """
    original_count = len(df)

    # Remove null/empty body
    df = df[df["body"].notna() & (df["body"].str.strip() != "")]
    after_empty = len(df)

    # Remove deleted/removed comments
    df = df[~df["body"].isin(REMOVED_MARKERS)]
    after_removed = len(df)

    # Remove bot authors
    df = df[~df["author"].isin(BOT_AUTHORS)]
    after_bots = len(df)

    # Remove bot commands (case-insensitive)
    body_lower = df["body"].str.lower().str.strip()
    is_bot_command = body_lower.isin(BOT_COMMANDS)
    df = df[~is_bot_command]
    after_commands = len(df)

    # Remove short comments
    df = df[df["body"].str.len() >= MIN_COMMENT_LENGTH]
    after_length = len(df)

    # Align to valid posts only (link_id is "t3_<post_id>")
    df["post_id"] = df["link_id"].str.replace("t3_", "", regex=False)
    df = df[df["post_id"].isin(valid_post_ids)]
    after_alignment = len(df)
    df = df.drop(columns=["post_id"])

    print(f"Comments cleaning summary:")
    print(f"  Original:              {original_count:>10,}")
    print(f"  After removing empty:  {after_empty:>10,} (-{original_count - after_empty:,})")
    print(f"  After removing del/rm: {after_removed:>10,} (-{after_empty - after_removed:,})")
    print(f"  After removing bots:   {after_bots:>10,} (-{after_removed - after_bots:,})")
    print(f"  After bot commands:    {after_commands:>10,} (-{after_bots - after_commands:,})")
    print(f"  After length filter:   {after_length:>10,} (-{after_commands - after_length:,})")
    print(f"  After post alignment:  {after_alignment:>10,} (-{after_length - after_alignment:,})")
    print(f"  Total removed:         {original_count - after_alignment:>10,} ({(original_count - after_alignment) / original_count * 100:.1f}%)")
    print()

    return df


def main():
    print("=" * 60)
    print("Reddit Data Cleaning Script")
    print("=" * 60)
    print()

    # Load data
    print("Loading posts...")
    posts_df = pd.read_parquet(POSTS_INPUT)
    print(f"Loaded {len(posts_df):,} posts\n")

    print("Loading comments...")
    comments_df = pd.read_parquet(COMMENTS_INPUT)
    print(f"Loaded {len(comments_df):,} comments\n")

    # Clean posts
    posts_cleaned = clean_posts(posts_df)

    # Get valid post IDs for alignment
    valid_post_ids = set(posts_cleaned["id"])

    # Clean comments (aligned to valid posts)
    comments_cleaned = clean_comments(comments_df, valid_post_ids)

    # Save cleaned data
    print("Saving cleaned data...")
    posts_cleaned.to_parquet(POSTS_OUTPUT, index=False)
    print(f"  Posts saved to: {POSTS_OUTPUT.name}")

    comments_cleaned.to_parquet(COMMENTS_OUTPUT, index=False)
    print(f"  Comments saved to: {COMMENTS_OUTPUT.name}")

    print()
    print("=" * 60)
    print("Done!")
    print(f"  Clean posts:    {len(posts_cleaned):>10,}")
    print(f"  Clean comments: {len(comments_cleaned):>10,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
