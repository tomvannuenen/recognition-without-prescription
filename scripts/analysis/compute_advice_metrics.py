#!/usr/bin/env python3
"""
Compute linguistic metrics for LLM advice vs human advice.

No LLM-as-judge: all metrics are descriptive, computed directly from text.
Measures concrete linguistic features that operationalize theoretical constructs
from appraisal theory (Engagement, Attitude, Graduation) and advice-giving style.

Usage:
    python compute_advice_metrics.py                  # Run on merged advice file
    python compute_advice_metrics.py --from-checkpoints  # Run directly from checkpoints

Output: data/advice_metrics/advice_metrics_<timestamp>.parquet
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
import spacy
from pathlib import Path
from datetime import datetime
from collections import Counter

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
ADVICE_DIR = DATA_DIR / "llm_advice"
OUTPUT_DIR = DATA_DIR / "advice_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load spaCy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# --- LEXICONS ---

HEDGE_WORDS = {
    'might', 'could', 'may', 'perhaps', 'possibly', 'maybe', 'probably',
    'seemingly', 'apparently', 'arguably', 'potentially', 'somewhat',
    'fairly', 'rather', 'quite', 'relatively',
}
HEDGE_PHRASES = [
    'it seems', 'it appears', 'i think', 'i believe', 'i feel',
    'in my opinion', 'not sure', 'not certain', 'hard to say',
    'it could be', 'it might be', 'it may be', 'from what you describe',
]

BOOSTER_WORDS = {
    'clearly', 'definitely', 'obviously', 'absolutely', 'certainly',
    'undoubtedly', 'always', 'never', 'completely', 'totally',
    'entirely', 'extremely', 'utterly', 'unquestionably',
}

DEONTIC_MODALS = {'should', 'must', 'ought', 'shall'}
DEONTIC_PHRASES = ['need to', 'have to', 'got to']
EPISTEMIC_MODALS = {'might', 'could', 'may', 'would', 'can'}

LEAVE_WORDS = [
    'leave', 'break up', 'breakup', 'divorce', 'end it', 'walk away',
    'dump', 'move on', 'get out', 'run', 'separate', 'left him',
    'left her', 'leaving', 'end the relationship', 'end this',
]
STAY_WORDS = [
    'stay', 'work on', 'communicate', 'work it out', 'couples therapy',
    'marriage counseling', 'give it time', 'work through', 'salvage',
    'reconcile', 'forgive', 'patience', 'compromise',
]

REDFLAG_WORDS = [
    'red flag', 'toxic', 'narcissist', 'gaslighting', 'manipulat',
    'abusive', 'controlling', 'predator', 'grooming', 'love bomb',
    'isolat', 'coercive', 'enabler',
]
THERAPY_WORDS = [
    'boundar', 'self-worth', 'therapy', 'therapist', 'self-care',
    'mental health', 'healing', 'trauma', 'self-esteem', 'codependen',
    'attachment', 'emotional intelligence', 'mindful', 'well-being',
    'counseling', 'psycholog',
]

INTENSIFIERS = {
    'very', 'extremely', 'incredibly', 'so', 'really', 'truly',
    'deeply', 'highly', 'seriously', 'absolutely', 'completely',
}
DOWNTONERS = {
    'somewhat', 'slightly', 'a bit', 'a little', 'kind of',
    'sort of', 'mildly', 'fairly', 'moderately', 'partly',
}


def count_pattern(text_lower: str, patterns: list) -> int:
    """Count occurrences of patterns in text."""
    return sum(text_lower.count(p) for p in patterns)


def compute_metrics(text: str) -> dict:
    """Compute all linguistic metrics for a single text."""
    if not text or len(text.strip()) < 50:
        return {"valid": False}

    text_lower = text.lower()
    doc = nlp(text)

    # --- LENGTH & STRUCTURE ---
    sentences = list(doc.sents)
    n_sentences = len(sentences)
    n_tokens = len([t for t in doc if not t.is_space])
    n_chars = len(text)
    n_paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    n_questions = text.count('?')
    n_exclamations = text.count('!')
    avg_sent_len = n_tokens / max(n_sentences, 1)

    # --- HEDGING & CERTAINTY (Engagement) ---
    hedge_count = sum(1 for t in doc if t.text.lower() in HEDGE_WORDS)
    hedge_count += count_pattern(text_lower, HEDGE_PHRASES)
    booster_count = sum(1 for t in doc if t.text.lower() in BOOSTER_WORDS)
    certainty_total = hedge_count + booster_count
    certainty_ratio = booster_count / certainty_total if certainty_total > 0 else 0.5

    # --- MODAL VERBS ---
    deontic_count = sum(1 for t in doc if t.text.lower() in DEONTIC_MODALS)
    deontic_count += count_pattern(text_lower, DEONTIC_PHRASES)
    epistemic_count = sum(1 for t in doc if t.text.lower() in EPISTEMIC_MODALS)
    modal_total = deontic_count + epistemic_count
    modal_ratio = deontic_count / modal_total if modal_total > 0 else 0.5

    # --- DIRECTIVENESS ---
    imperative_count = 0
    for sent in sentences:
        tokens = [t for t in sent if not t.is_space]
        if tokens and tokens[0].pos_ == 'VERB' and tokens[0].tag_ == 'VB':
            imperative_count += 1
    question_ratio = n_questions / max(n_sentences, 1)
    you_count = sum(1 for t in doc if t.text.lower() in {'you', 'your', 'yours', 'yourself'})
    you_density = you_count / max(n_sentences, 1)
    conditional_count = sum(1 for t in doc if t.text.lower() == 'if' and t.dep_ == 'mark')

    # --- PRONOUNS ---
    first_person = sum(1 for t in doc if t.text.lower() in {'i', 'me', 'my', 'mine', 'myself'})
    second_person = you_count
    third_person = sum(1 for t in doc if t.text.lower() in
                       {'he', 'she', 'they', 'him', 'her', 'his', 'them', 'their'})
    total_pronouns = first_person + second_person + third_person
    first_ratio = first_person / max(total_pronouns, 1)
    second_ratio = second_person / max(total_pronouns, 1)
    third_ratio = third_person / max(total_pronouns, 1)

    # --- RELATIONSHIP-SPECIFIC LEXICONS ---
    leave_count = count_pattern(text_lower, LEAVE_WORDS)
    stay_count = count_pattern(text_lower, STAY_WORDS)
    leave_stay_total = leave_count + stay_count
    leave_ratio = leave_count / leave_stay_total if leave_stay_total > 0 else 0.5

    redflag_count = count_pattern(text_lower, REDFLAG_WORDS)
    therapy_count = count_pattern(text_lower, THERAPY_WORDS)

    # --- GRADUATION (Intensification) ---
    intensifier_count = sum(1 for t in doc if t.text.lower() in INTENSIFIERS)
    downtoner_count = count_pattern(text_lower, list(DOWNTONERS))
    grad_total = intensifier_count + downtoner_count
    intensification_ratio = intensifier_count / grad_total if grad_total > 0 else 0.5

    # --- SENTIMENT (VADER) ---
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)

    return {
        "valid": True,
        # Length & structure
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "n_sentences": n_sentences,
        "avg_sent_len": avg_sent_len,
        "n_paragraphs": n_paragraphs,
        "n_questions": n_questions,
        "n_exclamations": n_exclamations,
        # Engagement (hedging/certainty)
        "hedge_count": hedge_count,
        "booster_count": booster_count,
        "certainty_ratio": certainty_ratio,
        # Modals
        "deontic_count": deontic_count,
        "epistemic_count": epistemic_count,
        "modal_ratio": modal_ratio,
        # Directiveness
        "imperative_count": imperative_count,
        "question_ratio": question_ratio,
        "you_density": you_density,
        "conditional_count": conditional_count,
        # Pronouns
        "first_person_ratio": first_ratio,
        "second_person_ratio": second_ratio,
        "third_person_ratio": third_ratio,
        # Relationship lexicons
        "leave_count": leave_count,
        "stay_count": stay_count,
        "leave_ratio": leave_ratio,
        "redflag_count": redflag_count,
        "therapy_count": therapy_count,
        # Graduation
        "intensifier_count": intensifier_count,
        "downtoner_count": downtoner_count,
        "intensification_ratio": intensification_ratio,
        # Sentiment
        "sentiment_compound": sentiment['compound'],
        "sentiment_pos": sentiment['pos'],
        "sentiment_neg": sentiment['neg'],
        "sentiment_neu": sentiment['neu'],
    }


def load_advice_from_checkpoints() -> pd.DataFrame:
    """Load advice responses from per-model checkpoint files."""
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


def load_human_comments() -> pd.DataFrame:
    """Load top human comment for each eligible post."""
    comments_df = pd.read_parquet(DATA_DIR / "r_relationship_advice_comments_cleaned.parquet")
    comments_df['post_id'] = comments_df['link_id'].str.replace('t3_', '', regex=False)

    # Quality filter (same as eligibility)
    quality = comments_df[
        (~comments_df['is_op']) &
        (comments_df['is_top_level'] == True) &
        (comments_df['score'] >= 5) &
        (comments_df['body'].str.len() >= 200)
    ]

    # Get top comment per post (highest score)
    top_comments = quality.sort_values('score', ascending=False).groupby('post_id').first()
    top_comments = top_comments[['body', 'score']].reset_index()
    top_comments.columns = ['post_id', 'human_response', 'human_score']

    return top_comments


def main():
    parser = argparse.ArgumentParser(description="Compute advice linguistic metrics")
    parser.add_argument("--from-checkpoints", action="store_true",
                        help="Load from checkpoint files instead of merged parquet")
    args = parser.parse_args()

    print("Loading advice data...")
    if args.from_checkpoints:
        advice_df = load_advice_from_checkpoints()
    else:
        # Load latest merged file
        merged_files = sorted(ADVICE_DIR.glob('llm_advice_*.parquet'))
        if not merged_files:
            print("No merged file found, loading from checkpoints...")
            advice_df = load_advice_from_checkpoints()
        else:
            advice_df = pd.read_parquet(merged_files[-1])

    print(f"  Posts with advice: {len(advice_df)}")

    # Load human comments
    print("Loading human comments...")
    human_df = load_human_comments()
    print(f"  Posts with human comments: {len(human_df)}")

    # Merge
    df = advice_df.merge(human_df, on='post_id', how='inner')
    print(f"  Posts with both: {len(df)}")

    # Normalize column names to short source names
    model_key_to_short = {
        'google_gemini-2-5-flash-lite': 'gemini',
        'deepseek_deepseek-chat-v3-0324': 'deepseek',
        'mistralai_ministral-8b': 'ministral',
        'openai_gpt-4-1-nano': 'gpt_nano',
    }
    rename_map = {}
    for col in df.columns:
        if col.endswith('_response'):
            key = col.replace('_response', '')
            short = model_key_to_short.get(key, key)
            if short != key:
                rename_map[col] = f'{short}_response'
    if rename_map:
        df = df.rename(columns=rename_map)

    # Identify response columns
    response_cols = [c for c in df.columns if c.endswith('_response')]
    sources = [c.replace('_response', '') for c in response_cols]
    print(f"  Sources: {sources}")

    # Compute metrics for each source
    print("\nComputing metrics...")
    all_metrics = []

    for idx, row in df.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(df)}...")

        post_id = row['post_id']

        for source in sources:
            text = row.get(f'{source}_response', '')
            if not text or not isinstance(text, str):
                continue
            metrics = compute_metrics(text)
            if metrics.get('valid'):
                metrics['post_id'] = post_id
                metrics['source'] = source
                del metrics['valid']
                all_metrics.append(metrics)

        # Human comment
        human_text = row.get('human_response', '')
        if human_text and isinstance(human_text, str):
            metrics = compute_metrics(human_text)
            if metrics.get('valid'):
                metrics['post_id'] = post_id
                metrics['source'] = 'human'
                del metrics['valid']
                all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n  Total metric rows: {len(metrics_df)}")
    print(f"  Per source:")
    for source, count in metrics_df['source'].value_counts().sort_index().items():
        print(f"    {source}: {count}")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f"advice_metrics_{timestamp}.parquet"
    metrics_df.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Quick summary
    print("\n" + "=" * 60)
    print("QUICK COMPARISON (medians)")
    print("=" * 60)
    summary_cols = ['n_tokens', 'certainty_ratio', 'modal_ratio', 'you_density',
                    'leave_ratio', 'therapy_count', 'sentiment_compound']
    summary = metrics_df.groupby('source')[summary_cols].median()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
