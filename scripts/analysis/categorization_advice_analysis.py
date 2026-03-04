"""
Analysis: Categorization-Advice Relationship

4 analyses exploring how model harm classifications relate to advice patterns:
A1: Does harm classification predict directive advice?
A2: Community leave-orientation vs model harm framing
A3: Model-specific perceptual patterns
A4: When models agree on harm, do they still diverge from community?
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load data
print("Loading data...")
assignments = pd.read_parquet('/Users/tomvannuenen/Library/CloudStorage/Dropbox/GitHub/DEV/advice/data/multi_model_assignment/multi_model_assignments_20260122_215645.parquet')
advice = pd.read_parquet('/Users/tomvannuenen/Library/CloudStorage/Dropbox/GitHub/DEV/advice/data/advice_metrics/advice_metrics_20260125_105648.parquet')

# Define harm-related topics
HARM_TOPICS = ['controlling_abusive_behavior', 'boundary_violations', 'infidelity_cheating', 'stalking_harassment_safety']

# Model column mapping
MODEL_COLS = {
    'gpt': 'openai-gpt-4-1-nano_primary_topic',
    'ministral': 'mistralai-ministral-8b_primary_topic',
    'deepseek': 'deepseek-deepseek-chat-v3-0324_primary_topic',
    'gemini': 'gemini-2-5-flash-lite_primary'
}

# Create harm flags for each model
models = list(MODEL_COLS.keys())
for model in models:
    col = MODEL_COLS[model]
    assignments[f'{model}_harm'] = assignments[col].isin(HARM_TOPICS)

# Count agreement levels
assignments['harm_count'] = assignments[[f'{m}_harm' for m in models]].sum(axis=1)
assignments['all_agree_harm'] = assignments['harm_count'] == 4
assignments['any_harm'] = assignments['harm_count'] >= 1
assignments['majority_harm'] = assignments['harm_count'] >= 3

print(f"\nTotal posts: {len(assignments)}")
print(f"All 4 models agree harm: {assignments['all_agree_harm'].sum()} ({100*assignments['all_agree_harm'].mean():.1f}%)")
print(f"Majority (3+) see harm: {assignments['majority_harm'].sum()} ({100*assignments['majority_harm'].mean():.1f}%)")
print(f"At least 1 sees harm: {assignments['any_harm'].sum()} ({100*assignments['any_harm'].mean():.1f}%)")

# Merge with advice metrics
# First, understand advice structure
print("\n" + "="*60)
print("ADVICE DATA STRUCTURE")
print("="*60)
print(f"Advice rows: {len(advice)}")
print(f"Unique post_ids: {advice['post_id'].nunique()}")
print(f"Sources: {advice['source'].unique()}")

# Separate human and LLM advice
human_advice = advice[advice['source'] == 'human'].copy()
llm_advice = advice[advice['source'] != 'human'].copy()

print(f"\nHuman advice rows: {len(human_advice)}")
print(f"LLM advice rows: {len(llm_advice)}")
print(f"LLM sources: {llm_advice['source'].unique()}")

# Key metrics to aggregate
METRICS = ['leave_ratio', 'certainty_ratio', 'hedge_count', 'therapy_count', 'deontic_count', 'n_tokens']

# Aggregate human advice per post
human_agg = human_advice.groupby('post_id').agg({m: 'mean' for m in METRICS}).reset_index()
human_agg.columns = ['post_id'] + [f'human_{c}' for c in human_agg.columns[1:]]

# Aggregate LLM advice per post (average across all LLM sources)
llm_agg = llm_advice.groupby('post_id').agg({m: 'mean' for m in METRICS}).reset_index()
llm_agg.columns = ['post_id'] + [f'llm_{c}' for c in llm_agg.columns[1:]]

# Merge all together
merged = assignments.merge(human_agg, on='post_id', how='left')
merged = merged.merge(llm_agg, on='post_id', how='left')

print(f"\nMerged dataset: {len(merged)} posts")
print(f"Posts with human advice: {merged['human_leave_ratio'].notna().sum()}")
print(f"Posts with LLM advice: {merged['llm_leave_ratio'].notna().sum()}")

# ============================================================
print("\n" + "="*60)
print("ANALYSIS 1: Does harm classification predict directive advice?")
print("="*60)

def compare_groups(df, group_col, metric_col, group_name):
    """Compare metric between harm and non-harm groups"""
    harm = df[df[group_col]][metric_col].dropna()
    no_harm = df[~df[group_col]][metric_col].dropna()

    if len(harm) < 10 or len(no_harm) < 10:
        return None

    t_stat, p_val = stats.ttest_ind(harm, no_harm)
    effect_size = (harm.mean() - no_harm.mean()) / np.sqrt((harm.std()**2 + no_harm.std()**2) / 2)

    return {
        'group': group_name,
        'harm_mean': harm.mean(),
        'no_harm_mean': no_harm.mean(),
        'diff': harm.mean() - no_harm.mean(),
        'effect_size': effect_size,
        'p_value': p_val,
        'n_harm': len(harm),
        'n_no_harm': len(no_harm)
    }

print("\n--- When models classify as harm, is human advice more directive? ---")
results = []
for model in models:
    for metric in ['human_leave_ratio', 'human_certainty_ratio', 'human_deontic_count']:
        r = compare_groups(merged, f'{model}_harm', metric, f'{model}_{metric}')
        if r:
            results.append(r)

if results:
    df_results = pd.DataFrame(results)
    print("\nHuman advice metrics by model harm classification:")
    print(df_results.to_string())

print("\n--- When models classify as harm, is LLM advice more directive? ---")
results = []
for model in models:
    for metric in ['llm_leave_ratio', 'llm_certainty_ratio', 'llm_deontic_count']:
        r = compare_groups(merged, f'{model}_harm', metric, f'{model}_{metric}')
        if r:
            results.append(r)

if results:
    df_results = pd.DataFrame(results)
    print("\nLLM advice metrics by model harm classification:")
    print(df_results.to_string())

# ============================================================
print("\n" + "="*60)
print("ANALYSIS 2: Community leave-orientation vs model harm framing")
print("="*60)

# Define high-leave posts (community strongly recommends leaving)
leave_threshold = 0.5
merged['high_leave'] = merged['human_leave_ratio'] >= leave_threshold

print(f"\nHigh-leave posts (human leave_ratio >= {leave_threshold}): {merged['high_leave'].sum()}")

# For high-leave posts, how often do models classify as harm?
high_leave = merged[merged['high_leave']]
print(f"\nOf {len(high_leave)} high-leave posts:")
for model in models:
    rate = high_leave[f'{model}_harm'].mean() * 100
    print(f"  {model} classifies as harm: {rate:.1f}%")

print(f"  All 4 models agree harm: {high_leave['all_agree_harm'].mean()*100:.1f}%")
print(f"  Majority (3+) see harm: {high_leave['majority_harm'].mean()*100:.1f}%")

# Conversely, for harm-classified posts, what's the leave ratio?
print("\n--- Leave ratios for different harm agreement levels ---")
for harm_level in ['any_harm', 'majority_harm', 'all_agree_harm']:
    subset = merged[merged[harm_level]]
    if len(subset) > 0:
        human_leave = subset['human_leave_ratio'].mean()
        llm_leave = subset['llm_leave_ratio'].mean()
        print(f"{harm_level}: human_leave={human_leave:.3f}, llm_leave={llm_leave:.3f}, gap={human_leave-llm_leave:.3f}")

# ============================================================
print("\n" + "="*60)
print("ANALYSIS 3: Model-specific perceptual patterns")
print("="*60)

# Which model is best at predicting community leave-orientation?
print("\n--- Correlation between model harm classification and human leave_ratio ---")
for model in models:
    valid = merged[[f'{model}_harm', 'human_leave_ratio']].dropna()
    corr, p = stats.pointbiserialr(valid[f'{model}_harm'], valid['human_leave_ratio'])
    print(f"{model}: r={corr:.3f}, p={p:.4f}")

# Unique perceptions - cases where one model disagrees
print("\n--- Unique perceptions (model sees harm when others don't) ---")
for model in models:
    # Model sees harm but no other model does
    unique = merged[(merged[f'{model}_harm']) & (merged['harm_count'] == 1)]
    n_unique = len(unique)
    pct = n_unique / merged[f'{model}_harm'].sum() * 100 if merged[f'{model}_harm'].sum() > 0 else 0
    print(f"{model}: {n_unique} unique harm classifications ({pct:.1f}% of its harm calls)")

# ============================================================
print("\n" + "="*60)
print("ANALYSIS 4: When models agree on harm, do they still diverge?")
print("="*60)

# This is the key test: on unanimous harm cases, do LLMs still hedge?
unanimous = merged[merged['all_agree_harm']].copy()
print(f"\nAnalyzing {len(unanimous)} posts where all 4 models agree on harm classification")

if len(unanimous) > 0:
    print("\n--- Advice metrics comparison (unanimous harm cases) ---")
    metrics = ['leave_ratio', 'certainty_ratio', 'hedge_count', 'therapy_count', 'deontic_count']

    for metric in metrics:
        human_col = f'human_{metric}'
        llm_col = f'llm_{metric}'

        valid = unanimous[[human_col, llm_col]].dropna()
        if len(valid) < 10:
            continue

        human_mean = valid[human_col].mean()
        llm_mean = valid[llm_col].mean()
        t_stat, p_val = stats.ttest_rel(valid[human_col], valid[llm_col])

        print(f"\n{metric}:")
        print(f"  Human: {human_mean:.3f}")
        print(f"  LLM:   {llm_mean:.3f}")
        print(f"  Gap:   {human_mean - llm_mean:.3f}")
        print(f"  p-value: {p_val:.4f}")

    # Compare to non-harm posts
    print("\n--- Comparing gaps: unanimous harm vs non-harm posts ---")
    non_harm = merged[~merged['any_harm']].copy()

    for metric in ['leave_ratio', 'certainty_ratio', 'therapy_count']:
        human_col = f'human_{metric}'
        llm_col = f'llm_{metric}'

        harm_valid = unanimous[[human_col, llm_col]].dropna()
        non_valid = non_harm[[human_col, llm_col]].dropna()

        if len(harm_valid) < 10 or len(non_valid) < 10:
            continue

        harm_gap = harm_valid[human_col].mean() - harm_valid[llm_col].mean()
        non_gap = non_valid[human_col].mean() - non_valid[llm_col].mean()

        print(f"\n{metric}:")
        print(f"  Gap on unanimous harm: {harm_gap:.3f}")
        print(f"  Gap on non-harm:       {non_gap:.3f}")
        print(f"  Difference:            {harm_gap - non_gap:.3f}")

# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# Key finding: Even when all models agree situation involves harm,
# LLMs still diverge from community in predictable ways
if len(unanimous) > 0:
    valid = unanimous[['human_leave_ratio', 'llm_leave_ratio']].dropna()
    if len(valid) > 0:
        human_leave = valid['human_leave_ratio'].mean()
        llm_leave = valid['llm_leave_ratio'].mean()
        print(f"\nOn {len(unanimous)} posts where all 4 models unanimously classify as harm:")
        print(f"  Human leave ratio: {human_leave:.3f}")
        print(f"  LLM leave ratio:   {llm_leave:.3f}")
        print(f"  Gap persists:      {human_leave - llm_leave:.3f}")

        # Calculate if gap is larger on harm posts
        non_valid = non_harm[['human_leave_ratio', 'llm_leave_ratio']].dropna()
        if len(non_valid) > 0:
            non_gap = non_valid['human_leave_ratio'].mean() - non_valid['llm_leave_ratio'].mean()
            print(f"\nFor comparison, on non-harm posts:")
            print(f"  Human leave ratio: {non_valid['human_leave_ratio'].mean():.3f}")
            print(f"  LLM leave ratio:   {non_valid['llm_leave_ratio'].mean():.3f}")
            print(f"  Gap:               {non_gap:.3f}")
            print(f"\nGap WIDENS on harm posts by: {(human_leave - llm_leave) - non_gap:.3f}")

print("\n" + "="*60)
print("DONE")
print("="*60)
