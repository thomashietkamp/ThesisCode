import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
import argparse
import sys
from pathlib import Path
import itertools


def load_votes_data(csv_path):
    """Load and parse the votes CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} total votes from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)


def separate_votes_by_method(df):
    """Separate votes by method: human, AI Gemini, AI DeepSeek, and AI GPT."""
    # AI Gemini votes have user_session starting with 'ai_gemini'
    ai_gemini_votes = df[df['user_session'].str.startswith(
        'ai_gemini', na=False)]

    # AI DeepSeek votes have user_session starting with 'ai_deepseek'
    ai_deepseek_votes = df[df['user_session'].str.startswith(
        'ai_deepseek', na=False)]

    # AI GPT votes have user_session starting with 'ai_gpt'
    ai_gpt_votes = df[df['user_session'].str.startswith(
        'ai_gpt', na=False)]

    # Human votes are everything else (not starting with 'ai_')
    human_votes = df[~df['user_session'].str.startswith('ai_', na=False)]

    print(f"Human votes: {len(human_votes)}")
    print(f"AI Gemini votes: {len(ai_gemini_votes)}")
    print(f"AI DeepSeek votes: {len(ai_deepseek_votes)}")
    print(f"AI GPT votes: {len(ai_gpt_votes)}")

    return {
        'human': human_votes,
        'ai_gemini': ai_gemini_votes,
        'ai_deepseek': ai_deepseek_votes,
        'ai_gpt': ai_gpt_votes
    }


def create_vote_key(row):
    """Create a unique key for each contract-pair combination."""
    return f"{row['contract_id']}|{row['pair_identifier']}"


def get_method_votes_dict(votes_df):
    """Convert votes dataframe to a dictionary keyed by contract-pair."""
    votes_dict = {}

    for _, row in votes_df.iterrows():
        key = create_vote_key(row)
        if key not in votes_dict:
            votes_dict[key] = []
        votes_dict[key].append(row['winner_key'])

    return votes_dict


def get_matched_votes_for_methods(method1_votes, method2_votes, method1_name, method2_name):
    """Find votes where both methods evaluated the same contract-pair."""
    method1_dict = get_method_votes_dict(method1_votes)
    method2_dict = get_method_votes_dict(method2_votes)

    matched_pairs = []

    for key in method1_dict.keys():
        if key in method2_dict:
            # For simplicity, use the most common vote for each method
            # (in case there are multiple votes for the same pair)
            method1_votes_for_pair = method1_dict[key]
            method2_votes_for_pair = method2_dict[key]

            # Get most common vote (mode) or first vote if tie
            method1_vote = max(set(method1_votes_for_pair),
                               key=method1_votes_for_pair.count)
            method2_vote = max(set(method2_votes_for_pair),
                               key=method2_votes_for_pair.count)

            matched_pairs.append({
                'pair_key': key,
                f'{method1_name}_vote': method1_vote,
                f'{method2_name}_vote': method2_vote,
                f'{method1_name}_count': len(method1_votes_for_pair),
                f'{method2_name}_count': len(method2_votes_for_pair)
            })

    return matched_pairs


def calculate_kappa_for_pair(matched_pairs, method1_name, method2_name):
    """Calculate Cohen's kappa coefficient for a pair of methods."""
    if not matched_pairs:
        print(
            f"No matching pairs found between {method1_name} and {method2_name} votes!")
        return None, None, None

    method1_labels = [pair[f'{method1_name}_vote'] for pair in matched_pairs]
    method2_labels = [pair[f'{method2_name}_vote'] for pair in matched_pairs]

    # Calculate kappa
    kappa = cohen_kappa_score(method1_labels, method2_labels)

    # Calculate simple agreement rate
    agreements = sum(1 for m1, m2 in zip(
        method1_labels, method2_labels) if m1 == m2)
    agreement_rate = agreements / len(matched_pairs)

    return kappa, agreement_rate, len(matched_pairs)


def print_detailed_analysis_for_pair(matched_pairs, method1_name, method2_name):
    """Print detailed analysis of the votes for a pair of methods."""
    print("\n" + "="*60)
    print(
        f"DETAILED ANALYSIS: {method1_name.upper()} vs {method2_name.upper()}")
    print("="*60)

    # Count vote distributions
    method1_vote_counts = defaultdict(int)
    method2_vote_counts = defaultdict(int)
    agreement_by_category = defaultdict(list)

    for pair in matched_pairs:
        method1_vote = pair[f'{method1_name}_vote']
        method2_vote = pair[f'{method2_name}_vote']

        method1_vote_counts[method1_vote] += 1
        method2_vote_counts[method2_vote] += 1

        if method1_vote == method2_vote:
            agreement_by_category[method1_vote].append(True)
        else:
            agreement_by_category[method1_vote].append(False)
            agreement_by_category[method2_vote].append(False)

    print(f"\n{method1_name.title()} vote distribution:")
    for vote, count in sorted(method1_vote_counts.items()):
        percentage = (count / len(matched_pairs)) * 100
        print(f"  {vote}: {count} ({percentage:.1f}%)")

    print(f"\n{method2_name.title()} vote distribution:")
    for vote, count in sorted(method2_vote_counts.items()):
        percentage = (count / len(matched_pairs)) * 100
        print(f"  {vote}: {count} ({percentage:.1f}%)")

    # Agreement analysis
    print(f"\nAgreement by category:")
    for category in sorted(set(list(method1_vote_counts.keys()) + list(method2_vote_counts.keys()))):
        agreements = agreement_by_category[category]
        if agreements:
            agreement_rate = sum(agreements) / len(agreements) * 100
            print(
                f"  {category}: {agreement_rate:.1f}% agreement ({sum(agreements)}/{len(agreements)})")


def print_confusion_matrix_for_pair(matched_pairs, method1_name, method2_name):
    """Print a confusion matrix for a pair of methods."""
    print("\n" + "="*40)
    print("CONFUSION MATRIX")
    print("="*40)
    print(f"Rows: {method1_name.title()}, Columns: {method2_name.title()}\n")

    # Get all unique labels
    all_labels = sorted(set([pair[f'{method1_name}_vote'] for pair in matched_pairs] +
                            [pair[f'{method2_name}_vote'] for pair in matched_pairs]))

    # Create confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for pair in matched_pairs:
        matrix[pair[f'{method1_name}_vote']][pair[f'{method2_name}_vote']] += 1

    # Print header
    print(f"{method1_name}\\{method2_name}", end="")
    for method2_label in all_labels:
        print(f"\t{method2_label}", end="")
    print()

    # Print matrix
    for method1_label in all_labels:
        print(f"{method1_label}", end="")
        for method2_label in all_labels:
            print(f"\t{matrix[method1_label][method2_label]}", end="")
        print()


def print_disagreement_examples(matched_pairs, method1_name, method2_name):
    """Print examples of disagreements between two methods."""
    print("\n" + "="*60)
    print(
        f"EXAMPLE DISAGREEMENTS: {method1_name.upper()} vs {method2_name.upper()}")
    print("="*60)

    disagreements = [
        pair for pair in matched_pairs
        if pair[f'{method1_name}_vote'] != pair[f'{method2_name}_vote']
    ]

    if disagreements:
        print(
            f"Found {len(disagreements)} disagreements out of {len(matched_pairs)} total pairs")
        print("\nFirst 5 disagreements:")
        for i, pair in enumerate(disagreements[:5]):
            contract_id = pair['pair_key'].split('|')[0]
            pair_id = pair['pair_key'].split('|')[1]
            print(f"{i+1}. Contract: {contract_id}")
            print(f"   Pair: {pair_id}")
            print(
                f"   {method1_name.title()} chose: {pair[f'{method1_name}_vote']}")
            print(
                f"   {method2_name.title()} chose: {pair[f'{method2_name}_vote']}")
            print()
    else:
        print("No disagreements found - perfect agreement!")


def interpret_kappa(kappa):
    """Interpret Cohen's kappa score."""
    if kappa < 0:
        return "Poor (worse than random)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Cohen\'s kappa between human, AI Gemini, AI DeepSeek, and AI GPT votes')
    parser.add_argument('csv_file', nargs='?',
                        default='src/stage2/data/votes_export_20250618_205707.csv',
                        help='Path to the CSV file containing votes')

    args = parser.parse_args()

    print("Cohen's Kappa Calculator for Multi-Method Voting Agreement")
    print("="*60)
    print("Comparing: Human vs AI Gemini vs AI DeepSeek vs AI GPT")
    print("="*60)

    # Load data
    df = load_votes_data(args.csv_file)

    # Separate votes by method
    votes_by_method = separate_votes_by_method(df)

    # Check if we have data for all methods
    methods_with_data = [method for method,
                         votes in votes_by_method.items() if len(votes) > 0]

    if len(methods_with_data) < 2:
        print("Error: Need at least 2 methods with votes for comparison!")
        sys.exit(1)

    print(f"\nMethods with data: {', '.join(methods_with_data)}")

    # Calculate pairwise comparisons
    method_pairs = list(itertools.combinations(methods_with_data, 2))

    print(f"\nPerforming {len(method_pairs)} pairwise comparisons...")

    # Store results for summary
    results_summary = []

    for method1, method2 in method_pairs:
        print(f"\n{'='*80}")
        print(f"COMPARISON: {method1.upper()} vs {method2.upper()}")
        print(f"{'='*80}")

        # Find matching votes
        matched_pairs = get_matched_votes_for_methods(
            votes_by_method[method1],
            votes_by_method[method2],
            method1,
            method2
        )

        if not matched_pairs:
            print(
                f"No matching contract-pair combinations found between {method1} and {method2}!")
            continue

        print(f"Found {len(matched_pairs)} matching contract-pair evaluations")

        # Calculate Cohen's kappa
        kappa, agreement_rate, n_pairs = calculate_kappa_for_pair(
            matched_pairs, method1, method2)

        # Store results
        results_summary.append({
            'methods': f"{method1} vs {method2}",
            'n_pairs': n_pairs,
            'agreement_rate': agreement_rate,
            'kappa': kappa,
            'interpretation': interpret_kappa(kappa)
        })

        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Number of matching pairs: {n_pairs}")
        print(
            f"Simple agreement rate: {agreement_rate:.3f} ({agreement_rate*100:.1f}%)")
        print(f"Cohen's kappa: {kappa:.3f}")
        print(f"Kappa interpretation: {interpret_kappa(kappa)}")

        # Detailed analysis
        print_detailed_analysis_for_pair(matched_pairs, method1, method2)
        print_confusion_matrix_for_pair(matched_pairs, method1, method2)
        print_disagreement_examples(matched_pairs, method1, method2)

    # Summary of all comparisons
    print("\n" + "="*80)
    print("SUMMARY OF ALL COMPARISONS")
    print("="*80)

    print(f"{'Comparison':<25} {'N Pairs':<10} {'Agreement':<12} {'Kappa':<8} {'Interpretation'}")
    print("-" * 80)

    for result in results_summary:
        agreement_pct = f"{result['agreement_rate']:.1%}"
        print(f"{result['methods']:<25} {result['n_pairs']:<10} "
              f"{agreement_pct:<12} {result['kappa']:<8.3f} {result['interpretation']}")

    # Find best and worst agreements
    if results_summary:
        best_kappa = max(results_summary, key=lambda x: x['kappa'])
        worst_kappa = min(results_summary, key=lambda x: x['kappa'])

        print(
            f"\nHighest agreement: {best_kappa['methods']} (κ = {best_kappa['kappa']:.3f})")
        print(
            f"Lowest agreement: {worst_kappa['methods']} (κ = {worst_kappa['kappa']:.3f})")


if __name__ == "__main__":
    main()
