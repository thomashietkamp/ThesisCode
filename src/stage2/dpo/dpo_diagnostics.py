#!/usr/bin/env python3
"""
DPO Training Diagnostics Script

This script helps diagnose issues with DPO training by analyzing:
1. Dataset quality and statistics
2. Model logprob distributions
3. Training stability metrics
4. Preference alignment

Usage:
    python dpo_diagnostics.py --dataset_path src/stage2/data/dpo_training_data.jsonl
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer
import argparse
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPODiagnostics:
    def __init__(self, dataset_path: str, model_name: str = "Qwen/Qwen3-14B"):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        self.data = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """Load and validate dataset"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        data = []

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")

        logger.info(f"Loaded {len(data)} samples")
        return data

    def analyze_dataset_statistics(self) -> Dict:
        """Analyze basic dataset statistics"""
        logger.info("Analyzing dataset statistics...")

        stats = {
            'total_samples': len(self.data),
            'prompt_lengths': [],
            'plus_trajectory_lengths': [],
            'minus_trajectory_lengths': [],
            'field_completeness': {}
        }

        required_fields = [
            'prompt', 'coord_think_plus', 'coord_out_plus',
            'reason_think_plus', 'reason_out_plus', 'valid_think_plus', 'valid_out_plus',
            'coord_think_minus', 'coord_out_minus', 'reason_think_minus',
            'reason_out_minus', 'valid_think_minus', 'valid_out_minus'
        ]

        for field in required_fields:
            stats['field_completeness'][field] = 0

        for item in self.data:
            # Check field completeness
            for field in required_fields:
                if field in item and item[field] and len(item[field].strip()) > 0:
                    stats['field_completeness'][field] += 1

            # Analyze lengths
            if 'prompt' in item:
                stats['prompt_lengths'].append(len(item['prompt']))

            # Plus trajectory
            plus_traj = (
                item.get('coord_think_plus', '') + item.get('coord_out_plus', '') +
                item.get('reason_think_plus', '') + item.get('reason_out_plus', '') +
                item.get('valid_think_plus', '') +
                item.get('valid_out_plus', '')
            )
            stats['plus_trajectory_lengths'].append(len(plus_traj))

            # Minus trajectory
            minus_traj = (
                item.get('coord_think_minus', '') + item.get('coord_out_minus', '') +
                item.get('reason_think_minus', '') + item.get('reason_out_minus', '') +
                item.get('valid_think_minus', '') +
                item.get('valid_out_minus', '')
            )
            stats['minus_trajectory_lengths'].append(len(minus_traj))

        # Convert to percentages
        for field in required_fields:
            stats['field_completeness'][field] = (
                stats['field_completeness'][field] / len(self.data) * 100
            )

        return stats

    def analyze_tokenization_stats(self) -> Dict:
        """Analyze tokenization statistics"""
        logger.info("Analyzing tokenization statistics...")

        stats = {
            'prompt_token_lengths': [],
            'plus_total_token_lengths': [],
            'minus_total_token_lengths': [],
            'truncation_warnings': 0
        }

        max_length = 6144  # Updated to match new training length

        for item in self.data:
            # Tokenize prompt
            prompt_tokens = self.tokenizer(
                item.get('prompt', ''),
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            stats['prompt_token_lengths'].append(
                prompt_tokens['input_ids'].shape[1])

            # Plus trajectory
            plus_full = (
                f"[PROMPT]{item.get('prompt', '')}"
                f"[COORDINATOR THINKING]{item.get('coord_think_plus', '')}"
                f"[COORDINATOR OUTPUT]{item.get('coord_out_plus', '')}"
                f"[REASONER THINKING]{item.get('reason_think_plus', '')}"
                f"[REASONER OUTPUT]{item.get('reason_out_plus', '')}"
                f"[VALIDATOR THINKING]{item.get('valid_think_plus', '')}"
                f"[VALIDATOR OUTPUT]{item.get('valid_out_plus', '')}"
            )

            plus_tokens = self.tokenizer(
                plus_full,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            plus_len = plus_tokens['input_ids'].shape[1]
            stats['plus_total_token_lengths'].append(plus_len)

            # Minus trajectory
            minus_full = (
                f"[PROMPT]{item.get('prompt', '')}"
                f"[COORDINATOR THINKING]{item.get('coord_think_minus', '')}"
                f"[COORDINATOR OUTPUT]{item.get('coord_out_minus', '')}"
                f"[REASONER THINKING]{item.get('reason_think_minus', '')}"
                f"[REASONER OUTPUT]{item.get('reason_out_minus', '')}"
                f"[VALIDATOR THINKING]{item.get('valid_think_minus', '')}"
                f"[VALIDATOR OUTPUT]{item.get('valid_out_minus', '')}"
            )

            minus_tokens = self.tokenizer(
                minus_full,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            minus_len = minus_tokens['input_ids'].shape[1]
            stats['minus_total_token_lengths'].append(minus_len)

            # Check for truncation
            if plus_len >= max_length or minus_len >= max_length:
                stats['truncation_warnings'] += 1

        return stats

    def analyze_preference_quality(self) -> Dict:
        """Analyze the quality of preference labels"""
        logger.info("Analyzing preference quality...")

        stats = {
            'length_differences': [],
            'content_similarities': [],
            'diversity_scores': []
        }

        for item in self.data:
            # Compare trajectory lengths
            plus_len = len(
                item.get('coord_think_plus', '') + item.get('coord_out_plus', '') +
                item.get('reason_think_plus', '') + item.get('reason_out_plus', '') +
                item.get('valid_think_plus', '') +
                item.get('valid_out_plus', '')
            )
            minus_len = len(
                item.get('coord_think_minus', '') + item.get('coord_out_minus', '') +
                item.get('reason_think_minus', '') + item.get('reason_out_minus', '') +
                item.get('valid_think_minus', '') +
                item.get('valid_out_minus', '')
            )

            stats['length_differences'].append(plus_len - minus_len)

            # Simple content similarity (Jaccard similarity)
            plus_words = set((
                item.get('coord_out_plus', '') + ' ' +
                item.get('reason_out_plus', '') + ' ' +
                item.get('valid_out_plus', '')
            ).lower().split())

            minus_words = set((
                item.get('coord_out_minus', '') + ' ' +
                item.get('reason_out_minus', '') + ' ' +
                item.get('valid_out_minus', '')
            ).lower().split())

            if len(plus_words) > 0 and len(minus_words) > 0:
                intersection = len(plus_words.intersection(minus_words))
                union = len(plus_words.union(minus_words))
                similarity = intersection / union if union > 0 else 0
                stats['content_similarities'].append(similarity)

        return stats

    def generate_report(self) -> str:
        """Generate a comprehensive diagnostic report"""
        logger.info("Generating diagnostic report...")

        dataset_stats = self.analyze_dataset_statistics()
        token_stats = self.analyze_tokenization_stats()
        preference_stats = self.analyze_preference_quality()

        report = []
        report.append("=" * 80)
        report.append("DPO TRAINING DIAGNOSTICS REPORT")
        report.append("=" * 80)
        report.append("")

        # Dataset overview
        report.append("📊 DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total samples: {dataset_stats['total_samples']}")
        report.append(
            f"Prompt length: {np.mean(dataset_stats['prompt_lengths']):.1f} ± {np.std(dataset_stats['prompt_lengths']):.1f} chars")
        report.append(
            f"Plus trajectory length: {np.mean(dataset_stats['plus_trajectory_lengths']):.1f} ± {np.std(dataset_stats['plus_trajectory_lengths']):.1f} chars")
        report.append(
            f"Minus trajectory length: {np.mean(dataset_stats['minus_trajectory_lengths']):.1f} ± {np.std(dataset_stats['minus_trajectory_lengths']):.1f} chars")
        report.append("")

        # Field completeness
        report.append("📋 FIELD COMPLETENESS")
        report.append("-" * 40)
        for field, percentage in dataset_stats['field_completeness'].items():
            status = "✅" if percentage >= 95 else "⚠️" if percentage >= 80 else "❌"
            report.append(f"{status} {field}: {percentage:.1f}%")
        report.append("")

        # Tokenization stats
        report.append("🔤 TOKENIZATION STATISTICS")
        report.append("-" * 40)
        report.append(
            f"Prompt tokens: {np.mean(token_stats['prompt_token_lengths']):.1f} ± {np.std(token_stats['prompt_token_lengths']):.1f}")
        report.append(
            f"Plus trajectory tokens: {np.mean(token_stats['plus_total_token_lengths']):.1f} ± {np.std(token_stats['plus_total_token_lengths']):.1f}")
        report.append(
            f"Minus trajectory tokens: {np.mean(token_stats['minus_total_token_lengths']):.1f} ± {np.std(token_stats['minus_total_token_lengths']):.1f}")
        report.append(
            f"Truncation warnings: {token_stats['truncation_warnings']} samples ({token_stats['truncation_warnings']/len(self.data)*100:.1f}%)")
        report.append("")

        # Preference quality
        report.append("🎯 PREFERENCE QUALITY")
        report.append("-" * 40)
        if preference_stats['length_differences']:
            avg_length_diff = np.mean(preference_stats['length_differences'])
            report.append(
                f"Length bias (plus - minus): {avg_length_diff:.1f} ± {np.std(preference_stats['length_differences']):.1f} chars")

            if abs(avg_length_diff) > 100:
                report.append(
                    "⚠️  Large length bias detected - may cause model to prefer based on length")
            else:
                report.append("✅ Length bias within acceptable range")

        if preference_stats['content_similarities']:
            avg_similarity = np.mean(preference_stats['content_similarities'])
            report.append(
                f"Content similarity: {avg_similarity:.3f} ± {np.std(preference_stats['content_similarities']):.3f}")

            if avg_similarity > 0.8:
                report.append(
                    "⚠️  High content similarity - preferences may be too subtle")
            elif avg_similarity < 0.3:
                report.append("✅ Good content diversity between preferences")
            else:
                report.append(
                    "✅ Moderate content similarity - good for learning")

        report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)

        if token_stats['truncation_warnings'] > len(self.data) * 0.1:
            report.append(
                "• Consider increasing max_length or shortening trajectories")

        if any(pct < 95 for pct in dataset_stats['field_completeness'].values()):
            report.append(
                "• Some fields have missing data - consider data cleaning")

        if preference_stats['length_differences'] and abs(np.mean(preference_stats['length_differences'])) > 100:
            report.append(
                "• Large length bias detected - consider length normalization")

        report.append(
            "• Use lower learning rate (1e-5) and alpha (0.1) for stability")
        report.append("• Monitor gradient norms and clip at 0.5")
        report.append("• Use warmup schedule for stable training")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="DPO Training Diagnostics")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to DPO training data JSONL file")
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B",
                        help="Model name for tokenizer")
    parser.add_argument("--output_file", help="Save report to file")

    args = parser.parse_args()

    # Run diagnostics
    diagnostics = DPODiagnostics(args.dataset_path, args.model_name)
    report = diagnostics.generate_report()

    # Print report
    print(report)

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        print(f"\n📁 Report saved to {args.output_file}")


if __name__ == "__main__":
    main()
