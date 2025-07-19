#!/usr/bin/env python3
"""
Human Evaluation Statistical Analysis Script

This script analyzes the three-way voting results from threeway_votes.json
and performs statistical testing and visualization for publication.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, friedmanchisquare, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 14

# TOGGLE: Set to True to include single stage, False to exclude it
INCLUDE_SINGLE_STAGE = False


def load_data(filename='threeway_votes.json'):
    """Load and parse the voting data from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    # Extract field names and values
    fields = data['fields']
    values = data['values']

    # Create DataFrame
    df = pd.DataFrame(values, columns=fields)

    # Convert score columns to numeric
    score_columns = [
        'fine_tuned_committee_clarity', 'fine_tuned_committee_legal',
        'fine_tuned_committee_reasoning', 'fine_tuned_committee_alignment',
        'single_stage_clarity', 'single_stage_legal',
        'single_stage_reasoning', 'single_stage_alignment',
        'non_fine_tuned_committee_clarity', 'non_fine_tuned_committee_legal',
        'non_fine_tuned_committee_reasoning', 'non_fine_tuned_committee_alignment'
    ]

    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def prepare_analysis_data(df, include_single_stage=True):
    """Prepare data for statistical analysis."""
    # Define models based on toggle
    if include_single_stage:
        models = ['fine_tuned_committee',
                  'single_stage', 'non_fine_tuned_committee']
    else:
        models = ['fine_tuned_committee', 'non_fine_tuned_committee']

    criteria = ['clarity', 'legal', 'reasoning', 'alignment']

    analysis_data = []

    for idx, row in df.iterrows():
        for model in models:
            for criterion in criteria:
                col_name = f"{model}_{criterion}"
                score = row[col_name]

                analysis_data.append({
                    'session_id': row['user_session'],
                    'contract_id': row['contract_id'],
                    'model': model,
                    'criterion': criterion,
                    'score': score,
                    'winner': row['winner_source']
                })

    analysis_df = pd.DataFrame(analysis_data)

    # Create model labels for better visualization
    model_labels = {
        'fine_tuned_committee': 'Fine-tuned Committee',
        'single_stage': 'Single Stage',
        'non_fine_tuned_committee': 'Non-fine-tuned Committee'
    }
    analysis_df['model_label'] = analysis_df['model'].map(model_labels)

    return analysis_df


def statistical_tests(analysis_df, include_single_stage=True):
    """Perform statistical tests on the data."""
    models = analysis_df['model'].unique()
    n_models = len(models)

    print("=" * 80)
    print("STATISTICAL ANALYSIS RESULTS")
    if include_single_stage:
        print("(Including Single Stage Model)")
    else:
        print("(Excluding Single Stage Model - Committee Comparison Only)")
    print("=" * 80)

    results = {}

    # 1. Overall model comparison across all criteria
    print(f"\n1. OVERALL MODEL PERFORMANCE COMPARISON ({n_models} models)")
    print("-" * 50)

    # Calculate overall scores per model (handle duplicates by taking mean)
    overall_scores = analysis_df.groupby(['session_id', 'model'])[
        'score'].mean().reset_index()
    pivot_overall = overall_scores.pivot(
        index='session_id', columns='model', values='score')

    # Check if we have enough data for statistical tests
    n_sessions = len(pivot_overall)
    print(f"Number of evaluation sessions: {n_sessions}")

    if n_sessions < 2:
        print(
            "Warning: Need at least 2 evaluation sessions for meaningful statistical tests")
        results['friedman_overall'] = None
        results['pairwise_overall'] = {}
    else:
        # Choose appropriate test based on number of models
        if n_models >= 3:
            # Friedman test (non-parametric repeated measures)
            try:
                model_data = [pivot_overall[model].dropna()
                              for model in models]
                friedman_stat, friedman_p = friedmanchisquare(*model_data)
                print(
                    f"Friedman Test: χ² = {friedman_stat:.4f}, p = {friedman_p:.4f}")
                results['friedman_overall'] = {
                    'statistic': friedman_stat, 'p_value': friedman_p}
            except Exception as e:
                print(f"Friedman test failed: {e}")
                results['friedman_overall'] = None
        elif n_models == 2:
            # Wilcoxon test for two models
            try:
                model1, model2 = models
                data1 = pivot_overall[model1].dropna()
                data2 = pivot_overall[model2].dropna()
                stat, p_val = wilcoxon(data1, data2, alternative='two-sided')
                print(f"Wilcoxon Test: W = {stat:.4f}, p = {p_val:.4f}")
                results['friedman_overall'] = {
                    'statistic': stat, 'p_value': p_val, 'test_type': 'wilcoxon'}
            except Exception as e:
                print(f"Wilcoxon test failed: {e}")
                results['friedman_overall'] = None

        # Pairwise comparisons using Wilcoxon signed-rank test
        pairs = []
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Avoid duplicates
                    pairs.append((model1, model2))

        print("\nPairwise comparisons (Wilcoxon signed-rank test):")
        results['pairwise_overall'] = {}
        for model1, model2 in pairs:
            try:
                data1 = pivot_overall[model1].dropna()
                data2 = pivot_overall[model2].dropna()
                if len(data1) >= 2 and len(data2) >= 2:
                    stat, p_val = wilcoxon(
                        data1, data2, alternative='two-sided')
                    print(f"{model1} vs {model2}: W = {stat:.4f}, p = {p_val:.4f}")
                    results['pairwise_overall'][f"{model1}_vs_{model2}"] = {
                        'statistic': stat, 'p_value': p_val}
                else:
                    print(
                        f"{model1} vs {model2}: Insufficient data for Wilcoxon test")
                    results['pairwise_overall'][f"{model1}_vs_{model2}"] = None
            except Exception as e:
                print(f"Wilcoxon test failed for {model1} vs {model2}: {e}")
                results['pairwise_overall'][f"{model1}_vs_{model2}"] = None

    # 2. Per-criterion analysis
    print("\n2. PER-CRITERION ANALYSIS")
    print("-" * 50)

    results['per_criterion'] = {}
    for criterion in ['clarity', 'legal', 'reasoning', 'alignment']:
        print(f"\n{criterion.upper()}:")
        criterion_data = analysis_df[analysis_df['criterion'] == criterion]

        # Handle duplicates by taking mean before pivoting
        criterion_scores = criterion_data.groupby(['session_id', 'model'])[
            'score'].mean().reset_index()

        try:
            pivot_criterion = criterion_scores.pivot(
                index='session_id', columns='model', values='score')
        except ValueError as e:
            print(f"  Pivot failed for {criterion}: {e}")
            # Alternative approach: use pivot_table which handles duplicates
            pivot_criterion = criterion_data.pivot_table(
                index='session_id',
                columns='model',
                values='score',
                aggfunc='mean'
            )

        if n_sessions < 2:
            print(
                f"  Insufficient data for statistical tests ({n_sessions} sessions)")
            results['per_criterion'][criterion] = {
                'friedman': None, 'pairwise': {}}
            continue

        # Choose appropriate test based on number of models
        if n_models >= 3:
            # Friedman test for this criterion
            try:
                model_data = [pivot_criterion[model].dropna()
                              for model in models]
                friedman_stat, friedman_p = friedmanchisquare(*model_data)
                print(
                    f"  Friedman Test: χ² = {friedman_stat:.4f}, p = {friedman_p:.4f}")
                results['per_criterion'][criterion] = {
                    'friedman': {'statistic': friedman_stat, 'p_value': friedman_p},
                    'pairwise': {}
                }
            except Exception as e:
                print(f"  Friedman test failed: {e}")
                results['per_criterion'][criterion] = {
                    'friedman': None, 'pairwise': {}}
        elif n_models == 2:
            # Wilcoxon test for two models
            try:
                model1, model2 = models
                data1 = pivot_criterion[model1].dropna()
                data2 = pivot_criterion[model2].dropna()
                stat, p_val = wilcoxon(data1, data2, alternative='two-sided')
                print(f"  Wilcoxon Test: W = {stat:.4f}, p = {p_val:.4f}")
                results['per_criterion'][criterion] = {
                    'friedman': {'statistic': stat, 'p_value': p_val, 'test_type': 'wilcoxon'},
                    'pairwise': {}
                }
            except Exception as e:
                print(f"  Wilcoxon test failed: {e}")
                results['per_criterion'][criterion] = {
                    'friedman': None, 'pairwise': {}}

        # Pairwise comparisons for this criterion
        for model1, model2 in pairs:
            try:
                data1 = pivot_criterion[model1].dropna()
                data2 = pivot_criterion[model2].dropna()
                if len(data1) >= 2 and len(data2) >= 2:
                    stat, p_val = wilcoxon(
                        data1, data2, alternative='two-sided')
                    print(
                        f"  {model1} vs {model2}: W = {stat:.4f}, p = {p_val:.4f}")
                    results['per_criterion'][criterion]['pairwise'][f"{model1}_vs_{model2}"] = {
                        'statistic': stat, 'p_value': p_val
                    }
                else:
                    print(
                        f"  {model1} vs {model2}: Insufficient data for Wilcoxon test")
                    results['per_criterion'][criterion]['pairwise'][f"{model1}_vs_{model2}"] = None
            except Exception as e:
                print(f"  Wilcoxon test failed for {model1} vs {model2}: {e}")
                results['per_criterion'][criterion]['pairwise'][f"{model1}_vs_{model2}"] = None

    # 3. Winner analysis
    print("\n3. WINNER PREFERENCE ANALYSIS")
    print("-" * 50)

    # Use unique sessions only for winner analysis
    winner_data = analysis_df.drop_duplicates(subset=['session_id'])

    # Filter winner data based on include_single_stage toggle
    if not include_single_stage:
        winner_data = winner_data[winner_data['winner'] != 'single_stage']

    winner_counts = winner_data['winner'].value_counts()
    print("Winner frequencies:")
    for winner, count in winner_counts.items():
        print(f"  {winner}: {count}")

    # Chi-square test for winner preferences
    if len(winner_counts) > 1 and len(winner_data) > 0:
        try:
            # Equal preference expected
            expected_freq = len(winner_data) / len(winner_counts)
            chi2_stat = sum((count - expected_freq)**2 /
                            expected_freq for count in winner_counts)
            chi2_p = 1 - stats.chi2.cdf(chi2_stat, len(winner_counts) - 1)
            print(
                f"\nChi-square test for equal preference: χ² = {chi2_stat:.4f}, p = {chi2_p:.4f}")
            results['winner_preference'] = {
                'counts': winner_counts.to_dict(),
                'chi2_statistic': chi2_stat,
                'chi2_p_value': chi2_p
            }
        except Exception as e:
            print(f"Chi-square test failed: {e}")
            results['winner_preference'] = {
                'counts': winner_counts.to_dict(),
                'chi2_statistic': None,
                'chi2_p_value': None
            }
    else:
        print("Insufficient data for chi-square test")
        results['winner_preference'] = {
            'counts': winner_counts.to_dict() if len(winner_counts) > 0 else {},
            'chi2_statistic': None,
            'chi2_p_value': None
        }

    # 4. Descriptive statistics
    print("\n4. DESCRIPTIVE STATISTICS")
    print("-" * 50)

    # Group by model and criterion, then aggregate (handle duplicates)
    desc_stats = analysis_df.groupby(['model', 'criterion'])['score'].agg(
        ['count', 'mean', 'std', 'median']).round(3)
    print(desc_stats)
    results['descriptive_stats'] = desc_stats.to_dict()

    return results


def create_plots(analysis_df, save_plots=True, include_single_stage=True):
    """Create comprehensive visualizations."""
    print("\nGenerating plots...")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_style("whitegrid")

    # Adjust colors based on number of models - using KMPG brand palette from improved_process_results.py
    if include_single_stage:
        # KMPG blue, Pink, Light blue
        colors = ['#00338D', '#FD349C', '#76D2FF']
        plot_suffix = ""
    else:
        colors = ['#00338D', '#76D2FF']  # KMPG blue, Light blue (skip pink)
        plot_suffix = "_committee_only"

    # 1. Box plot of scores by model and criterion
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=analysis_df, x='criterion', y='score',
                hue='model_label', palette=colors)
    plt.title('Score Distribution by Model and Criterion',
              fontsize=20, fontweight='bold')
    plt.xlabel('Evaluation Criterion', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.legend(title='Model Type', title_fontsize=14, fontsize=13)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(
            f'score_distribution_boxplot{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Violin plot for more detailed distribution
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=analysis_df, x='criterion', y='score',
                   hue='model_label', palette=colors)
    plt.title('Score Distribution Density by Model and Criterion',
              fontsize=20, fontweight='bold')
    plt.xlabel('Evaluation Criterion', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.legend(title='Model Type', title_fontsize=14, fontsize=13)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(
            f'score_distribution_violin{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Mean scores comparison
    mean_scores = analysis_df.groupby(['model_label', 'criterion'])[
        'score'].mean().reset_index()
    pivot_means = mean_scores.pivot(
        index='criterion', columns='model_label', values='score')

    plt.figure(figsize=(12, 8))
    ax = pivot_means.plot(kind='bar', color=colors, width=0.8)
    plt.title('Mean Scores by Model and Criterion',
              fontsize=20, fontweight='bold')
    plt.xlabel('Evaluation Criterion', fontsize=16)
    plt.ylabel('Mean Score', fontsize=16)
    plt.legend(title='Model Type', title_fontsize=14, fontsize=13,
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', alpha=0.7)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            f'mean_scores_comparison{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Overall performance radar chart
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw=dict(projection='polar'))

    criteria = ['clarity', 'legal', 'reasoning', 'alignment']
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

    models = analysis_df['model_label'].unique()
    for i, model in enumerate(models):
        model_data = analysis_df[analysis_df['model_label'] == model]
        means = [model_data[model_data['criterion'] == c]['score'].mean()
                 for c in criteria]
        means += [means[0]]  # Complete the circle

        ax.plot(angles, means, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, means, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in criteria], fontsize=14)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=12)
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=13)
    plt.title('Model Performance Comparison (Radar Chart)',
              size=20, fontweight='bold', pad=20)

    if save_plots:
        plt.savefig(
            f'performance_radar_chart{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. Winner preference pie chart
    winner_data = analysis_df.drop_duplicates(subset=['session_id'])
    if not include_single_stage:
        winner_data = winner_data[winner_data['winner'] != 'single_stage']

    winner_counts = winner_data['winner'].value_counts()

    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%',
                                       colors=colors[:len(winner_counts)], startangle=90, textprops={'fontsize': 14})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    plt.title('Winner Preference Distribution', fontsize=20, fontweight='bold')
    plt.axis('equal')

    if save_plots:
        plt.savefig(
            f'winner_preference_pie{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. Heatmap of mean scores
    plt.figure(figsize=(10, 6))
    heatmap_data = analysis_df.groupby(['model_label', 'criterion'])[
        'score'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=3,
                fmt='.2f', cbar_kws={'label': 'Mean Score'}, linewidths=0.5)
    plt.title('Mean Scores Heatmap', fontsize=20, fontweight='bold')
    plt.xlabel('Evaluation Criterion', fontsize=16)
    plt.ylabel('Model Type', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)

    if save_plots:
        plt.savefig(
            f'mean_scores_heatmap{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 7. Statistical significance summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot means with error bars for each criterion
    criteria = ['clarity', 'legal', 'reasoning', 'alignment']

    for i, criterion in enumerate(criteria):
        ax = [ax1, ax2, ax3, ax4][i]
        criterion_data = analysis_df[analysis_df['criterion'] == criterion]

        means = []
        stds = []
        model_labels = criterion_data['model_label'].unique()

        for model in model_labels:
            model_scores = criterion_data[criterion_data['model_label']
                                          == model]['score']
            means.append(model_scores.mean())
            stds.append(model_scores.std())

        x_pos = np.arange(len(model_labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(model_labels)], alpha=0.7,
                      error_kw={'elinewidth': 2, 'capthick': 2})

        ax.set_xlabel('Model Type', fontsize=22)
        ax.set_ylabel('Mean Score', fontsize=22)
        ax.set_title(f'{criterion.capitalize()} Scores',
                     fontsize=24, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(' ', '\n')
                           for m in model_labels], fontsize=20)
        ax.set_ylim(0, 5)
        ax.grid(axis='y', alpha=0.3)

        # Increase tick label font size
        ax.tick_params(axis='y', labelsize=20)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{mean:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=18)

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            f'criterion_comparison_bars{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results_summary(results, analysis_df, include_single_stage=True):
    """Save a comprehensive results summary to a text file."""
    filename = 'statistical_analysis_results.txt'
    if not include_single_stage:
        filename = 'statistical_analysis_results_committee_only.txt'

    with open(filename, 'w') as f:
        f.write("HUMAN EVALUATION STATISTICAL ANALYSIS RESULTS\n")
        if include_single_stage:
            f.write("(Including Single Stage Model)\n")
        else:
            f.write("(Committee Models Only - Single Stage Excluded)\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total evaluations: {len(analysis_df)}\n")
        f.write(f"Unique sessions: {analysis_df['session_id'].nunique()}\n")
        f.write(f"Models compared: {analysis_df['model'].nunique()}\n")
        f.write(
            f"Evaluation criteria: {analysis_df['criterion'].nunique()}\n\n")

        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-" * 30 + "\n")
        desc_stats = analysis_df.groupby(['model_label', 'criterion'])['score'].agg([
            'count', 'mean', 'std', 'min', 'max']).round(3)
        f.write(str(desc_stats))
        f.write("\n\n")

        f.write("STATISTICAL TEST RESULTS\n")
        f.write("-" * 30 + "\n")

        # Overall Friedman test
        if results.get('friedman_overall'):
            test_type = results['friedman_overall'].get(
                'test_type', 'friedman')
            if test_type == 'wilcoxon':
                f.write(
                    f"Overall Wilcoxon test: W = {results['friedman_overall']['statistic']:.4f}, ")
            else:
                f.write(
                    f"Overall Friedman test: χ² = {results['friedman_overall']['statistic']:.4f}, ")
            f.write(f"p = {results['friedman_overall']['p_value']:.4f}\n")
        else:
            f.write("Overall test: Not performed (insufficient data)\n")

        # Per-criterion tests
        f.write(f"\nPer-criterion tests:\n")
        for criterion, result in results.get('per_criterion', {}).items():
            if result.get('friedman'):
                test_type = result['friedman'].get('test_type', 'friedman')
                if test_type == 'wilcoxon':
                    f.write(
                        f"{criterion} (Wilcoxon): W = {result['friedman']['statistic']:.4f}, ")
                else:
                    f.write(
                        f"{criterion} (Friedman): χ² = {result['friedman']['statistic']:.4f}, ")
                f.write(f"p = {result['friedman']['p_value']:.4f}\n")
            else:
                f.write(f"{criterion}: Not performed (insufficient data)\n")

        # Pairwise comparisons
        f.write("\nPairwise comparisons (Wilcoxon signed-rank tests):\n")
        if results.get('pairwise_overall'):
            for comparison, result in results['pairwise_overall'].items():
                if result:
                    f.write(
                        f"Overall {comparison}: W = {result['statistic']:.4f}, p = {result['p_value']:.4f}\n")
                else:
                    f.write(
                        f"Overall {comparison}: Not performed (insufficient data)\n")

        # Winner preference analysis
        f.write(f"\nWinner preference analysis:\n")
        if results.get('winner_preference'):
            for winner, count in results['winner_preference']['counts'].items():
                f.write(f"{winner}: {count} selections\n")

            if results['winner_preference']['chi2_statistic'] is not None:
                f.write(
                    f"Chi-square test: χ² = {results['winner_preference']['chi2_statistic']:.4f}, ")
                f.write(
                    f"p = {results['winner_preference']['chi2_p_value']:.4f}\n")
            else:
                f.write(
                    "Chi-square test: Not performed (insufficient data or single category)\n")

        # Data adequacy notes
        f.write(f"\nDATA ADEQUACY NOTES\n")
        f.write("-" * 30 + "\n")
        n_sessions = analysis_df['session_id'].nunique()
        n_models = analysis_df['model'].nunique()

        f.write(
            f"Analysis mode: {'3-way comparison' if include_single_stage else '2-way committee comparison'}\n")
        f.write(f"Models in analysis: {n_models}\n")

        if n_sessions < 3:
            f.write(
                "WARNING: Very small sample size. Results should be interpreted with caution.\n")
            f.write(
                "Recommendation: Collect more evaluation data for robust statistical analysis.\n")
        elif n_sessions < 10:
            f.write(
                "CAUTION: Small sample size. Consider collecting more data for stronger conclusions.\n")
        else:
            f.write("Sample size is adequate for statistical analysis.\n")

        if n_sessions < 2:
            f.write(
                "Note: Many statistical tests require at least 2 evaluation sessions.\n")

        f.write(
            f"\nFor robust statistical conclusions, consider collecting at least 10-20 evaluation sessions.\n")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("HUMAN EVALUATION ANALYSIS")
    print(
        f"Mode: {'3-way comparison' if INCLUDE_SINGLE_STAGE else '2-way committee comparison'}")
    print("=" * 80)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} evaluation records")

    print("Preparing analysis data...")
    analysis_df = prepare_analysis_data(
        df, include_single_stage=INCLUDE_SINGLE_STAGE)
    print(f"Prepared {len(analysis_df)} data points for analysis")

    # Perform statistical tests
    results = statistical_tests(
        analysis_df, include_single_stage=INCLUDE_SINGLE_STAGE)

    # Create visualizations
    create_plots(analysis_df, include_single_stage=INCLUDE_SINGLE_STAGE)

    # Save comprehensive results
    save_results_summary(results, analysis_df,
                         include_single_stage=INCLUDE_SINGLE_STAGE)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    mode_suffix = "" if INCLUDE_SINGLE_STAGE else "_committee_only"
    print("Generated files:")
    print(f"- score_distribution_boxplot{mode_suffix}.png")
    print(f"- score_distribution_violin{mode_suffix}.png")
    print(f"- mean_scores_comparison{mode_suffix}.png")
    print(f"- performance_radar_chart{mode_suffix}.png")
    print(f"- winner_preference_pie{mode_suffix}.png")
    print(f"- mean_scores_heatmap{mode_suffix}.png")
    print(f"- criterion_comparison_bars{mode_suffix}.png")
    print(f"- statistical_analysis_results{mode_suffix}.txt")
    print("=" * 80)
    print(f"\nTo change analysis mode, edit INCLUDE_SINGLE_STAGE at line 19")
    print(f"Current setting: INCLUDE_SINGLE_STAGE = {INCLUDE_SINGLE_STAGE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
