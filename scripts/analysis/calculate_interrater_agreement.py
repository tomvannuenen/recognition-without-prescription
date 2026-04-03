#!/usr/bin/env python3
"""
Calculate Inter-Annotator Agreement for Pairwise Validation

Compares coding from two annotators on the pairwise validation task.
Reports DIRECTIONAL AGREEMENT: when both coders make a directional judgment
(human or llm), how often do they agree on direction?

Usage:
    python calculate_interrater_agreement.py

Requires:
    - data/pairwise_validation.csv (Coder 1)
    - data/validation_coder2.csv (Coder 2)

Output:
    - Directional agreement rates
    - Raw agreement rates
    - data/interrater_agreement_results.csv
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


def analyze_dimension(c1, c2, dim_name):
    """Analyze agreement for one dimension."""
    results = {}

    # Exclude where EITHER coder skipped
    valid_mask = (c1 != 'skip') & (c2 != 'skip')
    c1_valid = c1[valid_mask]
    c2_valid = c2[valid_mask]
    n_valid = len(c1_valid)

    # Raw agreement (including 'equal')
    raw_agreement = (c1_valid == c2_valid).mean() if n_valid > 0 else np.nan

    # Cohen's kappa
    try:
        kappa = cohen_kappa_score(c1_valid, c2_valid)
    except:
        kappa = np.nan

    # DIRECTIONAL AGREEMENT: when both make directional judgment (human or llm)
    directional_mask = (np.isin(c1_valid, ['human', 'llm'])) & (np.isin(c2_valid, ['human', 'llm']))
    n_directional = directional_mask.sum()

    if n_directional > 0:
        c1_dir = c1_valid[directional_mask]
        c2_dir = c2_valid[directional_mask]
        directional_agreement = (c1_dir == c2_dir).mean()
        n_agree_direction = (c1_dir == c2_dir).sum()
    else:
        directional_agreement = np.nan
        n_agree_direction = 0

    # Count disagreement types
    # Cases where one says human and other says llm (true directional disagreement)
    true_disagreement = ((c1_valid == 'human') & (c2_valid == 'llm')).sum() + \
                        ((c1_valid == 'llm') & (c2_valid == 'human')).sum()

    results = {
        'dimension': dim_name,
        'n_valid': n_valid,
        'raw_agreement': raw_agreement,
        'cohens_kappa': kappa,
        'n_directional': n_directional,
        'n_agree_direction': n_agree_direction,
        'directional_agreement': directional_agreement,
        'n_true_disagreement': true_disagreement
    }

    return results


def main():
    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("=" * 70)

    # Load both coders' data
    try:
        coder1 = pd.read_csv('data/pairwise_validation.csv')
        print(f"\nCoder 1 data loaded: {len(coder1)} items")
    except FileNotFoundError:
        print("ERROR: data/pairwise_validation.csv not found")
        return

    try:
        coder2 = pd.read_csv('data/validation_coder2.csv')
        print(f"Coder 2 data loaded: {len(coder2)} items")
    except FileNotFoundError:
        print("\nERROR: data/validation_coder2.csv not found")
        return

    # Merge on post_id to align items
    merged = coder1[['post_id', 'more_certain', 'more_leave_oriented', 'more_therapeutic']].merge(
        coder2[['post_id', 'more_certain', 'more_leave_oriented', 'more_therapeutic']],
        on='post_id',
        suffixes=('_c1', '_c2')
    )

    print(f"Matched items: {len(merged)}")

    dimensions = ['more_certain', 'more_leave_oriented', 'more_therapeutic']
    results = []

    for dim in dimensions:
        print(f"\n{'='*70}")
        print(f"DIMENSION: {dim.upper()}")
        print('='*70)

        c1 = merged[f'{dim}_c1']
        c2 = merged[f'{dim}_c2']

        r = analyze_dimension(c1.values, c2.values, dim)
        results.append(r)

        print(f"\nValid pairs (both non-skip): {r['n_valid']}")
        print(f"Raw agreement: {r['raw_agreement']:.1%}")
        print(f"Cohen's κ: {r['cohens_kappa']:.3f}")
        print(f"\nDIRECTIONAL (both say human or llm):")
        print(f"  Cases: {r['n_directional']}")
        print(f"  Agree on direction: {r['n_agree_direction']}/{r['n_directional']} = {r['directional_agreement']:.0%}")
        print(f"  True disagreements (human vs llm): {r['n_true_disagreement']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    print("\nDirectional agreement (when both coders made a directional judgment):")
    for _, row in results_df.iterrows():
        dim = row['dimension'].replace('more_', '')
        n = int(row['n_directional'])
        agree = int(row['n_agree_direction'])
        pct = row['directional_agreement'] * 100
        print(f"  {dim}: {agree}/{n} = {pct:.0f}%")

    print(f"\nTrue directional disagreements (one says human, other says llm):")
    for _, row in results_df.iterrows():
        dim = row['dimension'].replace('more_', '')
        print(f"  {dim}: {int(row['n_true_disagreement'])} cases")

    # Save results
    results_df.to_csv('data/interrater_agreement_results.csv', index=False)
    print(f"\nResults saved to: data/interrater_agreement_results.csv")


if __name__ == "__main__":
    main()
