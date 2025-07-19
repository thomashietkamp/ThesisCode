#!/usr/bin/env python3
"""
CUAD Dataset Statistics Calculator

This script calculates comprehensive statistics for the CUAD (Contract Understanding Atticus Dataset).
It analyzes contract lengths, token counts, clause frequencies, and various other metrics.

Usage:
    python cuad_statistics.py

Input:
    - data/CUAD_v1/CUAD_v1.json (CUAD dataset in SQuAD format)

Output:
    - Detailed statistics printed to console
    - JSON file with all statistics
    - Various plots saved to output_plots/
"""

import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import tiktoken
from datetime import datetime


def load_cuad_data(file_path: str) -> Dict[str, Any]:
    """Load CUAD dataset from JSON file."""
    print(f"Loading CUAD dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data['data'])} contracts")
    return data


def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens using tiktoken (OpenAI's tokenizer)."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to simple word count if tiktoken fails
        return len(text.split())


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in text using basic punctuation."""
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])


def count_paragraphs(text: str) -> int:
    """Count paragraphs in text."""
    paragraphs = text.split('\n\n')
    return len([p for p in paragraphs if p.strip()])


def count_pages(text: str, chars_per_page: int = 3000) -> float:
    """Estimate page count based on character count."""
    return len(text) / chars_per_page


def extract_dates_from_text(text: str) -> List[str]:
    """Extract dates from text using regex patterns."""
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
        r'\b\w+ \d{1,2}, \d{4}\b',     # Month DD, YYYY
        r'\b\d{1,2} \w+ \d{4}\b',      # DD Month YYYY
    ]

    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return dates


def analyze_contract_lengths(contracts: List[Dict]) -> Dict[str, Any]:
    """Analyze contract length statistics."""
    print("Analyzing contract lengths...")

    lengths = {
        'characters': [],
        'words': [],
        'pages': [],
        'paragraphs': [],
        'tokens_tiktoken': []
    }

    for contract in contracts:
        context = contract['paragraphs'][0]['context']

        lengths['characters'].append(len(context))
        lengths['words'].append(count_words(context))
        lengths['pages'].append(count_pages(context))
        lengths['paragraphs'].append(count_paragraphs(context))
        lengths['tokens_tiktoken'].append(count_tokens_tiktoken(context))

    stats = {}
    for metric, values in lengths.items():
        stats[metric] = {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'total': sum(values)
        }

    return stats, lengths


def analyze_clause_categories(contracts: List[Dict]) -> Dict[str, Any]:
    """Analyze clause category statistics."""
    print("Analyzing clause categories...")

    # Extract all questions (categories) and their statistics
    category_stats = defaultdict(lambda: {
        'total_questions': 0,
        'has_answer': 0,
        'is_impossible': 0,
        'answer_lengths': [],
        'answer_types': defaultdict(int)
    })

    all_questions = set()

    for contract in contracts:
        qas = contract['paragraphs'][0]['qas']

        for qa in qas:
            question = qa['question']
            all_questions.add(question)

            stats = category_stats[question]
            stats['total_questions'] += 1

            if qa['is_impossible']:
                stats['is_impossible'] += 1
            else:
                if qa['answers']:
                    stats['has_answer'] += 1
                    for answer in qa['answers']:
                        # Handle case where text might be a list or string
                        if isinstance(answer['text'], list):
                            answer_text = ' '.join(str(item)
                                                   for item in answer['text']).strip()
                        else:
                            answer_text = str(answer['text']).strip()
                        stats['answer_lengths'].append(len(answer_text))

                        # Classify answer types
                        if answer_text.lower() in ['yes', 'no']:
                            stats['answer_types']['yes_no'] += 1
                        elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', answer_text):
                            stats['answer_types']['date'] += 1
                        elif answer_text.lower() == 'perpetual':
                            stats['answer_types']['perpetual'] += 1
                        elif re.match(r'\d+', answer_text):
                            stats['answer_types']['numeric'] += 1
                        else:
                            stats['answer_types']['text'] += 1

    # Calculate percentages and summary stats
    processed_stats = {}
    for question, stats in category_stats.items():
        processed_stats[question] = {
            'total_contracts': len(contracts),
            'answered_contracts': stats['has_answer'],
            'impossible_contracts': stats['is_impossible'],
            'answer_rate': stats['has_answer'] / len(contracts) * 100,
            'impossible_rate': stats['is_impossible'] / len(contracts) * 100,
            'avg_answer_length': np.mean(stats['answer_lengths']) if stats['answer_lengths'] else 0,
            'answer_types': dict(stats['answer_types'])
        }

    return processed_stats, len(all_questions)


def analyze_temporal_patterns(contracts: List[Dict]) -> Dict[str, Any]:
    """Analyze temporal patterns in contracts."""
    print("Analyzing temporal patterns...")

    dates_found = []
    contract_years = []

    for contract in contracts:
        # Extract year from contract title if possible
        title = contract['title']
        year_match = re.search(r'(\d{4})', title)
        if year_match:
            year = int(year_match.group(1))
            if 1990 <= year <= 2030:  # Reasonable range
                contract_years.append(year)

        # Extract dates from contract text
        context = contract['paragraphs'][0]['context']
        dates = extract_dates_from_text(context)
        dates_found.extend(dates)

    return {
        'contract_years': {
            'count': len(contract_years),
            'min_year': min(contract_years) if contract_years else None,
            'max_year': max(contract_years) if contract_years else None,
            'year_distribution': Counter(contract_years)
        },
        'dates_in_text': {
            'total_dates_found': len(dates_found),
            'unique_dates': len(set(dates_found)),
            'sample_dates': dates_found[:10]  # First 10 dates as examples
        }
    }


def analyze_linguistic_features(contracts: List[Dict]) -> Dict[str, Any]:
    """Analyze linguistic features of contracts."""
    print("Analyzing linguistic features...")

    all_text = ""
    contract_complexities = []

    for contract in contracts:
        context = contract['paragraphs'][0]['context']
        all_text += context + " "

        # Calculate complexity metrics per contract
        words = count_words(context)
        sentences = count_sentences(context)
        avg_sentence_length = words / sentences if sentences > 0 else 0
        contract_complexities.append(avg_sentence_length)

    # Overall vocabulary analysis
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter(words)

    # Legal term patterns
    legal_terms = [
        'shall', 'agreement', 'party', 'parties', 'contract', 'clause',
        'termination', 'breach', 'liability', 'indemnify', 'hereby',
        'whereas', 'therefore', 'notwithstanding', 'covenant', 'warranty'
    ]

    legal_term_freq = {term: word_freq.get(term, 0) for term in legal_terms}

    return {
        'vocabulary': {
            'total_words': len(words),
            'unique_words': len(word_freq),
            'vocabulary_richness': len(word_freq) / len(words),
            'most_common_words': word_freq.most_common(20)
        },
        'complexity': {
            'avg_sentence_length_per_contract': {
                'mean': np.mean(contract_complexities),
                'std': np.std(contract_complexities),
                'min': min(contract_complexities),
                'max': max(contract_complexities)
            }
        },
        'legal_terms': legal_term_freq
    }


def create_visualizations(stats: Dict[str, Any], output_dir: str = "../../output_plots"):
    """Create various visualizations of the statistics."""
    print("Creating visualizations...")

    os.makedirs(output_dir, exist_ok=True)

    # Define custom color palette (RGB values converted to 0-1 scale)
    custom_colors = [
        (0/255, 51/255, 141/255),    # Dark blue
        (0/255, 94/255, 184/255),    # Medium blue
        (0/255, 145/255, 218/255),   # Light blue
        (72/255, 54/255, 152/255)    # Purple
    ]

    # Set modern style with larger fonts
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white'
    })

    # 1. Contract length distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Contract Length Distributions',
                 fontsize=22, fontweight='bold', y=0.98)

    length_metrics = ['characters', 'words', 'tokens_tiktoken', 'pages']
    for i, metric in enumerate(length_metrics):
        ax = axes[i//2, i % 2]
        values = stats['contract_lengths'][1][metric]

        # Create histogram with custom color
        n, bins, patches = ax.hist(values, bins=30, alpha=0.8,
                                   edgecolor='white', linewidth=1.5,
                                   color=custom_colors[i % len(custom_colors)])

        # Clean title and labels
        title = metric.replace("_", " ").replace("tiktoken", "GPT").title()
        ax.set_title(f'Distribution of {title}',
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel(title, fontsize=16, fontweight='medium')
        ax.set_ylabel('Frequency', fontsize=16, fontweight='medium')

        # Remove grid and clean up axes
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

        # Add subtle statistics annotation
        mean_val = np.mean(values)
        if metric == 'pages':
            ax.text(0.7, 0.9, f'Avg: {mean_val:.1f}', transform=ax.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor=custom_colors[i], alpha=0.2))
        else:
            ax.text(0.7, 0.9, f'Avg: {mean_val:,.0f}', transform=ax.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor=custom_colors[i], alpha=0.2))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/contract_length_distributions.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # 2. Clause category answer rates
    clause_stats = stats['clause_categories'][0]
    categories = list(clause_stats.keys())
    answer_rates = [clause_stats[cat]['answer_rate'] for cat in categories]

    plt.figure(figsize=(22, 12))
    bars = plt.barh(range(len(categories)), answer_rates,
                    color=[custom_colors[0] if rate > 80 else
                           custom_colors[1] if rate > 50 else
                           custom_colors[2] if rate > 20 else
                           custom_colors[3] for rate in answer_rates])

    plt.yticks(range(len(categories)), [
               cat[:60] + '...' if len(cat) > 60 else cat for cat in categories])
    plt.xlabel('Answer Rate (%)', fontsize=18, fontweight='medium')
    plt.title('Clause Category Answer Rates Across All Contracts',
              fontsize=20, fontweight='bold', pad=30)

    # Clean up axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/clause_answer_rates.png",
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 3. Contract years distribution
    if stats['temporal_patterns']['contract_years']['count'] > 0:
        year_dist = stats['temporal_patterns']['contract_years']['year_distribution']
        years = sorted(year_dist.keys())
        counts = [year_dist[year] for year in years]

        plt.figure(figsize=(14, 8))
        plt.plot(years, counts, marker='o', linewidth=3, markersize=8,
                 color=custom_colors[0], markerfacecolor=custom_colors[1])
        plt.title('Distribution of Contracts by Year',
                  fontsize=20, fontweight='bold', pad=30)
        plt.xlabel('Year', fontsize=18, fontweight='medium')
        plt.ylabel('Number of Contracts', fontsize=18, fontweight='medium')

        # Clean up axes
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/contract_years_distribution.png",
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    # 4. Answer type distribution
    answer_type_totals = defaultdict(int)
    for cat_stats in clause_stats.values():
        for answer_type, count in cat_stats['answer_types'].items():
            answer_type_totals[answer_type] += count

    if answer_type_totals:
        plt.figure(figsize=(12, 10))
        types = list(answer_type_totals.keys())
        counts = list(answer_type_totals.values())

        wedges, texts, autotexts = plt.pie(counts, labels=types, autopct='%1.1f%%',
                                           colors=custom_colors[:len(
                                               types)], startangle=90,
                                           textprops={'fontsize': 14})
        plt.title('Distribution of Answer Types Across All Categories',
                  fontsize=20, fontweight='bold', pad=30)

        # Enhance text visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/answer_types_distribution.png",
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    # 5. Clause count distribution (count total answer spans, not just questions)
    clause_counts_per_contract = []
    for contract in stats['raw_data']['data']:
        qas = contract['paragraphs'][0]['qas']
        clause_count = sum(
            len(qa['answers']) for qa in qas if qa['answers'] and not qa['is_impossible'])
        clause_counts_per_contract.append(clause_count)

    plt.figure(figsize=(14, 8))
    n, bins, patches = plt.hist(clause_counts_per_contract, bins=5, alpha=0.8,
                                edgecolor='white', linewidth=1.5,
                                color=custom_colors[0])

    plt.title('Distribution of Labeled Spans per Contract',
              fontsize=20, fontweight='bold', pad=30)
    plt.xlabel('Number of Labeled Spans per Contract',
               fontsize=18, fontweight='medium')
    plt.ylabel('Number of Contracts', fontsize=18, fontweight='medium')

    # Clean up axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.grid(False)

    # Increase font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add statistics annotation
    mean_clauses = np.mean(clause_counts_per_contract)
    median_clauses = np.median(clause_counts_per_contract)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/labeled_spans_distribution.png",
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"✓ Visualizations saved to {output_dir}/")


def print_summary_statistics(stats: Dict[str, Any]):
    """Print a summary of key statistics."""
    print("\n" + "="*80)
    print("CUAD DATASET STATISTICS SUMMARY")
    print("="*80)

    # Basic dataset info
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Total Contracts: {len(stats['raw_data']['data'])}")
    print(f"Total Categories: {stats['clause_categories'][1]}")
    print(f"Dataset Version: {stats['raw_data']['version']}")

    # Contract length statistics
    print(f"\n📏 CONTRACT LENGTH STATISTICS")
    length_stats = stats['contract_lengths'][0]
    print(f"Average Characters: {length_stats['characters']['mean']:,.0f}")
    print(f"Average Words: {length_stats['words']['mean']:,.0f}")
    print(
        f"Average Tokens (GPT): {length_stats['tokens_tiktoken']['mean']:,.0f}")
    print(f"Average Pages: {length_stats['pages']['mean']:,.1f}")
    print(
        f"Total Words (All Contracts): {length_stats['words']['total']:,.0f}")

    # Clause count statistics
    print(f"\n📋 LABELED SPAN STATISTICS")
    clause_counts = []
    for contract in stats['raw_data']['data']:
        qas = contract['paragraphs'][0]['qas']
        clause_count = sum(
            len(qa['answers']) for qa in qas if qa['answers'] and not qa['is_impossible'])
        clause_counts.append(clause_count)

    print(f"Average Labeled Spans per Contract: {np.mean(clause_counts):.1f}")
    print(f"Median Labeled Spans per Contract: {np.median(clause_counts):.1f}")
    print(f"Min Labeled Spans in a Contract: {min(clause_counts)}")
    print(f"Max Labeled Spans in a Contract: {max(clause_counts)}")
    print(f"Total Labeled Spans (All Contracts): {sum(clause_counts):,}")

    # Top answered categories
    print(f"\n🎯 TOP 10 MOST ANSWERED CATEGORIES")
    clause_stats = stats['clause_categories'][0]
    sorted_categories = sorted(clause_stats.items(),
                               key=lambda x: x[1]['answer_rate'], reverse=True)[:10]
    for i, (category, data) in enumerate(sorted_categories, 1):
        short_cat = category[:60] + "..." if len(category) > 60 else category
        print(f"{i:2d}. {short_cat:<63} {data['answer_rate']:5.1f}%")

    # Least answered categories
    print(f"\n🎯 TOP 10 LEAST ANSWERED CATEGORIES")
    sorted_categories = sorted(clause_stats.items(),
                               key=lambda x: x[1]['answer_rate'])[:10]
    for i, (category, data) in enumerate(sorted_categories, 1):
        short_cat = category[:60] + "..." if len(category) > 60 else category
        print(f"{i:2d}. {short_cat:<63} {data['answer_rate']:5.1f}%")

    # Temporal patterns
    print(f"\n📅 TEMPORAL PATTERNS")
    temporal = stats['temporal_patterns']
    if temporal['contract_years']['count'] > 0:
        print(
            f"Contracts with identifiable years: {temporal['contract_years']['count']}")
        print(
            f"Year range: {temporal['contract_years']['min_year']} - {temporal['contract_years']['max_year']}")
    print(
        f"Total dates found in text: {temporal['dates_in_text']['total_dates_found']:,}")

    # Linguistic features
    print(f"\n📝 LINGUISTIC FEATURES")
    linguistic = stats['linguistic_features']
    print(f"Total unique words: {linguistic['vocabulary']['unique_words']:,}")
    print(
        f"Vocabulary richness: {linguistic['vocabulary']['vocabulary_richness']:.4f}")
    print(
        f"Average sentence length: {linguistic['complexity']['avg_sentence_length_per_contract']['mean']:.1f} words")

    print(f"\n🏆 TOP 10 MOST COMMON WORDS")
    for i, (word, count) in enumerate(linguistic['vocabulary']['most_common_words'][:10], 1):
        print(f"{i:2d}. {word:<15} {count:,}")

    print("\n" + "="*80)


def save_detailed_statistics(stats: Dict[str, Any], output_file: str = "../../cuad_detailed_statistics.json"):
    """Save detailed statistics to JSON file."""
    print(f"Saving detailed statistics to {output_file}...")

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Add metadata
    stats_with_metadata = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script_version': '1.0',
            'description': 'Comprehensive statistics for CUAD dataset'
        },
        'statistics': convert_numpy_types(stats)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_with_metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Statistics saved to {output_file}")


def main():
    """Main function to run all analyses."""
    print("CUAD Dataset Statistics Calculator")
    print("=" * 50)

    # Load data
    data_path = "../../data/CUAD_v1/CUAD_v1.json"
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find dataset at {data_path}")
        print("Please ensure the CUAD dataset is available at the specified path.")
        return

    data = load_cuad_data(data_path)
    contracts = data['data']

    # Run all analyses
    print("\nRunning comprehensive analysis...")

    # Contract length analysis
    contract_length_stats, contract_lengths = analyze_contract_lengths(
        contracts)

    # Clause category analysis
    clause_category_stats, num_categories = analyze_clause_categories(
        contracts)

    # Temporal analysis
    temporal_stats = analyze_temporal_patterns(contracts)

    # Linguistic analysis
    linguistic_stats = analyze_linguistic_features(contracts)

    # Compile all statistics
    all_stats = {
        'raw_data': data,
        'contract_lengths': (contract_length_stats, contract_lengths),
        'clause_categories': (clause_category_stats, num_categories),
        'temporal_patterns': temporal_stats,
        'linguistic_features': linguistic_stats
    }

    # Create visualizations
    create_visualizations(all_stats)

    # Print summary
    print_summary_statistics(all_stats)

    # Save detailed statistics
    # Remove raw data before saving to avoid huge file
    stats_to_save = {k: v for k, v in all_stats.items() if k != 'raw_data'}
    stats_to_save['dataset_info'] = {
        'total_contracts': len(contracts),
        'version': data['version'],
        'total_categories': num_categories
    }
    save_detailed_statistics(stats_to_save)

    print("\n✅ Analysis complete!")
    print("Check output_plots/ for visualizations and cuad_detailed_statistics.json for detailed results.")


if __name__ == "__main__":
    main()
