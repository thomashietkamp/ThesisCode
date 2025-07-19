#!/usr/bin/env python3
"""
Preference Analysis Script for Stage 2 Data

This script analyzes preferences between three types of agents:
1. Random generated user sessions (human-like)
2. AI Gemini 
3. AI DeepSeek

The script analyzes voting patterns, length preferences, and other metrics
from the CSV voting data and JSON configuration files.
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PreferenceAnalyzer:
    def __init__(self, csv_path, json_dir):
        """Initialize the analyzer with data paths."""
        self.csv_path = csv_path
        self.json_dir = json_dir
        self.votes_df = None
        self.json_data = {}
        self.agent_types = {
            'human': 'Random User Sessions',
            'ai_gemini': 'AI Gemini',
            'ai_deepseek': 'AI DeepSeek'
        }

    def load_data(self):
        """Load and prepare all data."""
        print("Loading voting data...")
        self.votes_df = pd.read_csv(self.csv_path)

        print("Loading JSON configuration data...")
        self._load_json_data()

        print("Preprocessing data...")
        self._preprocess_data()

    def _load_json_data(self):
        """Load all JSON files and extract metadata."""
        json_files = [f for f in os.listdir(
            self.json_dir) if f.endswith('.json')]

        for file in json_files:
            file_path = os.path.join(self.json_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    contract_name = file.replace('.json', '')
                    self.json_data[contract_name] = data
            except Exception as e:
                print(f"Error loading {file}: {e}")

    def _preprocess_data(self):
        """Preprocess the voting data."""
        # Categorize sessions by type
        self.votes_df['agent_type'] = self.votes_df['user_session'].apply(
            self._categorize_session)

        # Extract numeric option keys for analysis
        self.votes_df['option1_num'] = pd.to_numeric(
            self.votes_df['option1_key'], errors='coerce')
        self.votes_df['option2_num'] = pd.to_numeric(
            self.votes_df['option2_key'], errors='coerce')
        self.votes_df['winner_num'] = pd.to_numeric(
            self.votes_df['winner_key'], errors='coerce')

        # Add model preference columns
        self.votes_df['preferred_model'] = self.votes_df.apply(
            self._get_preferred_model, axis=1)

    def _categorize_session(self, session_id):
        """Categorize session by agent type."""
        if pd.isna(session_id):
            return 'unknown'
        session_id = str(session_id).lower()

        if session_id.startswith('ai_gemini'):
            return 'ai_gemini'
        elif session_id.startswith('ai_deepseek'):
            return 'ai_deepseek'
        else:
            return 'human'

    def _get_preferred_model(self, row):
        """Determine which model was preferred in a vote."""
        if pd.isna(row['winner_num']) or row['winner_key'] == 'UNCLEAR':
            return 'unclear'

        # Map winner to model configuration
        winner = int(row['winner_num'])
        return self._map_option_to_model(winner, row['contract_id'])

    def _map_option_to_model(self, option_key, contract_id):
        """Map option key to model type based on JSON configuration."""
        if contract_id not in self.json_data:
            return 'unknown'

        config_data = self.json_data[contract_id]
        if str(option_key) not in config_data:
            return 'unknown'

        model_config = config_data[str(option_key)]['cfg']
        model_id = model_config.get('model_id', '')
        persona = model_config.get('persona', '')

        if 'deepseek' in model_id.lower():
            return f'deepseek_{persona}'
        elif 'qwen' in model_id.lower():
            return f'qwen_{persona}'
        else:
            return 'unknown'

    def analyze_overall_preferences(self):
        """Analyze overall model preferences by agent type."""
        print("\n" + "="*60)
        print("OVERALL MODEL PREFERENCES BY AGENT TYPE")
        print("="*60)

        # Create preference matrix
        preference_counts = pd.crosstab(
            self.votes_df['agent_type'],
            self.votes_df['preferred_model']
        )

        # Calculate percentages
        preference_pcts = preference_counts.div(
            preference_counts.sum(axis=1), axis=0) * 100

        print("\nRaw vote counts:")
        print(preference_counts)

        print("\nPercentages:")
        print(preference_pcts.round(2))

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Raw counts heatmap
        sns.heatmap(preference_counts, annot=True,
                    fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Raw Vote Counts by Agent Type and Preferred Model')
        ax1.set_xlabel('Preferred Model')
        ax1.set_ylabel('Agent Type')

        # Percentage heatmap
        sns.heatmap(preference_pcts, annot=True,
                    fmt='.1f', cmap='Reds', ax=ax2)
        ax2.set_title('Preference Percentages by Agent Type')
        ax2.set_xlabel('Preferred Model')
        ax2.set_ylabel('Agent Type')

        plt.tight_layout()
        plt.savefig('overall_preferences.png', dpi=300, bbox_inches='tight')
        plt.show()

        return preference_counts, preference_pcts

    def analyze_pairwise_preferences(self):
        """Analyze preferences in pairwise comparisons."""
        print("\n" + "="*60)
        print("PAIRWISE PREFERENCE ANALYSIS")
        print("="*60)

        # Focus on clear wins (exclude UNCLEAR)
        clear_votes = self.votes_df[self.votes_df['winner_key'] != 'UNCLEAR'].copy(
        )

        pairwise_stats = defaultdict(lambda: defaultdict(int))

        for _, row in clear_votes.iterrows():
            agent_type = row['agent_type']
            option1 = self._map_option_to_model(
                row['option1_num'], row['contract_id'])
            option2 = self._map_option_to_model(
                row['option2_num'], row['contract_id'])
            winner = row['preferred_model']

            if option1 != 'unknown' and option2 != 'unknown' and winner != 'unknown':
                pair_key = f"{option1}_vs_{option2}"
                pairwise_stats[agent_type][f"{pair_key}_total"] += 1
                if winner == option1:
                    pairwise_stats[agent_type][f"{pair_key}_{option1}_wins"] += 1
                else:
                    pairwise_stats[agent_type][f"{pair_key}_{option2}_wins"] += 1

        # Print summary
        for agent_type in pairwise_stats:
            print(f"\n{agent_type.upper()} Agent Pairwise Preferences:")
            agent_stats = pairwise_stats[agent_type]

            # Find unique pairs
            pairs = set()
            for key in agent_stats.keys():
                if '_total' in key:
                    pairs.add(key.replace('_total', ''))

            for pair in sorted(pairs):
                total = agent_stats.get(f"{pair}_total", 0)
                if total > 0:
                    print(f"  {pair}: {total} total votes")

                    # Find winners for this pair
                    for model in ['deepseek_neutral_legal', 'deepseek_risk_averse',
                                  'qwen_neutral_legal', 'qwen_risk_averse']:
                        wins = agent_stats.get(f"{pair}_{model}_wins", 0)
                        if wins > 0:
                            pct = (wins / total) * 100
                            print(f"    {model}: {wins} wins ({pct:.1f}%)")

    def analyze_model_complexity_preferences(self):
        """Analyze if agents prefer simpler or more complex model outputs."""
        print("\n" + "="*60)
        print("MODEL COMPLEXITY PREFERENCE ANALYSIS")
        print("="*60)

        # Analyze length preferences based on JSON data
        length_analysis = defaultdict(list)

        for contract_id, data in self.json_data.items():
            contract_votes = self.votes_df[self.votes_df['contract_id']
                                           == contract_id]

            for _, vote in contract_votes.iterrows():
                if vote['winner_key'] != 'UNCLEAR':
                    winner_key = str(int(vote['winner_num']))
                    if winner_key in data:
                        # Get the length of the final report
                        report_length = len(
                            data[winner_key].get('final_report', ''))
                        length_analysis[vote['agent_type']].append(
                            report_length)

        # Calculate statistics
        for agent_type in ['human', 'ai_gemini', 'ai_deepseek']:
            if agent_type in length_analysis:
                lengths = length_analysis[agent_type]
                if lengths:
                    mean_length = np.mean(lengths)
                    median_length = np.median(lengths)
                    std_length = np.std(lengths)

                    print(f"\n{agent_type.upper()} preferred report lengths:")
                    print(f"  Mean: {mean_length:.0f} characters")
                    print(f"  Median: {median_length:.0f} characters")
                    print(f"  Std Dev: {std_length:.0f} characters")
                    print(f"  Total votes: {len(lengths)}")

        # Create length distribution plot
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent_type in ['human', 'ai_gemini', 'ai_deepseek']:
            if agent_type in length_analysis and length_analysis[agent_type]:
                ax.hist(length_analysis[agent_type], alpha=0.7,
                        label=self.agent_types.get(agent_type, agent_type), bins=20)

        ax.set_xlabel('Report Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Preferred Report Lengths by Agent Type')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('length_preferences.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_persona_preferences(self):
        """Analyze preferences between neutral_legal and risk_averse personas."""
        print("\n" + "="*60)
        print("PERSONA PREFERENCE ANALYSIS")
        print("="*60)

        # Extract persona from preferred model
        self.votes_df['preferred_persona'] = self.votes_df['preferred_model'].apply(
            lambda x: 'neutral_legal' if 'neutral_legal' in str(x)
            else 'risk_averse' if 'risk_averse' in str(x)
            else 'other'
        )

        # Create persona preference crosstab
        persona_prefs = pd.crosstab(
            self.votes_df['agent_type'],
            self.votes_df['preferred_persona']
        )

        persona_pcts = persona_prefs.div(
            persona_prefs.sum(axis=1), axis=0) * 100

        print("\nPersona preference counts:")
        print(persona_prefs)

        print("\nPersona preference percentages:")
        print(persona_pcts.round(2))

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        persona_pcts.plot(kind='bar', ax=ax)
        ax.set_title('Persona Preferences by Agent Type')
        ax.set_xlabel('Agent Type')
        ax.set_ylabel('Percentage')
        ax.legend(title='Preferred Persona')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('persona_preferences.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_vote_patterns(self):
        """Analyze voting patterns and consistency."""
        print("\n" + "="*60)
        print("VOTING PATTERN ANALYSIS")
        print("="*60)

        # Analyze unclear votes
        unclear_rates = self.votes_df.groupby('agent_type')['winner_key'].apply(
            lambda x: (x == 'UNCLEAR').mean() * 100
        )

        print("Unclear vote rates by agent type:")
        for agent_type, rate in unclear_rates.items():
            print(f"  {agent_type}: {rate:.2f}%")

        # Analyze vote distribution
        vote_dist = self.votes_df['agent_type'].value_counts()

        print(f"\nTotal votes by agent type:")
        for agent_type, count in vote_dist.items():
            print(f"  {agent_type}: {count:,} votes")

        # Create summary visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Vote distribution
        vote_dist.plot(kind='bar', ax=ax1)
        ax1.set_title('Total Votes by Agent Type')
        ax1.set_xlabel('Agent Type')
        ax1.set_ylabel('Number of Votes')
        ax1.tick_params(axis='x', rotation=45)

        # Unclear rate
        unclear_rates.plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title('Unclear Vote Rate by Agent Type')
        ax2.set_xlabel('Agent Type')
        ax2.set_ylabel('Unclear Rate (%)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('voting_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PREFERENCE ANALYSIS SUMMARY")
        print("="*80)

        total_votes = len(self.votes_df)
        unique_contracts = self.votes_df['contract_id'].nunique()

        print(f"Dataset Overview:")
        print(f"  Total votes: {total_votes:,}")
        print(f"  Unique contracts: {unique_contracts}")
        print(f"  Unique sessions: {self.votes_df['user_session'].nunique()}")

        print(f"\nAgent Distribution:")
        agent_counts = self.votes_df['agent_type'].value_counts()
        for agent, count in agent_counts.items():
            pct = (count / total_votes) * 100
            print(f"  {agent}: {count:,} votes ({pct:.1f}%)")

        # Key findings
        print(f"\nKey Findings:")

        # Most preferred model overall
        model_prefs = self.votes_df[self.votes_df['preferred_model']
                                    != 'unclear']['preferred_model'].value_counts()
        if len(model_prefs) > 0:
            top_model = model_prefs.index[0]
            top_count = model_prefs.iloc[0]
            print(
                f"  Most preferred model overall: {top_model} ({top_count} votes)")

        # Agent-specific preferences
        for agent_type in ['human', 'ai_gemini', 'ai_deepseek']:
            agent_data = self.votes_df[
                (self.votes_df['agent_type'] == agent_type) &
                (self.votes_df['preferred_model'] != 'unclear')
            ]
            if len(agent_data) > 0:
                agent_prefs = agent_data['preferred_model'].value_counts()
                if len(agent_prefs) > 0:
                    top_pref = agent_prefs.index[0]
                    top_count = agent_prefs.iloc[0]
                    total_agent_votes = len(agent_data)
                    pct = (top_count / total_agent_votes) * 100
                    print(
                        f"  {agent_type} most prefers: {top_pref} ({pct:.1f}%)")


def main():
    """Main execution function."""
    # Define paths
    csv_path = "src/stage2/data/votes_export_20250617_143226.csv"
    json_dir = "src/stage2/data/stage2_out"

    # Initialize analyzer
    analyzer = PreferenceAnalyzer(csv_path, json_dir)

    # Load data
    analyzer.load_data()

    # Run all analyses
    print("Starting comprehensive preference analysis...")

    analyzer.analyze_overall_preferences()
    analyzer.analyze_pairwise_preferences()
    analyzer.analyze_model_complexity_preferences()
    analyzer.analyze_persona_preferences()
    analyzer.analyze_vote_patterns()
    analyzer.generate_summary_report()

    print("\nAnalysis complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
