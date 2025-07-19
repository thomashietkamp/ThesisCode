#!/usr/bin/env python3
"""
Script to create a professional training loss plot for DPO training results.
Uses the same styling as improved_process_results.py for consistency.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


def create_training_loss_plot():
    """
    Create a professional training loss plot using the provided training data.
    """
    # Set style for better aesthetics (same as improved_process_results.py)
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial',
                                       'DejaVu Sans', 'Liberation Sans']

    # Training loss data extracted from the user's output
    # Epoch 0 losses (selected representative batches)
    epoch_0_losses = [1.0559, 1.2487, 1.0538, 1.1423, 0.6800, 1.0694,
                      1.0458, 0.9186, 0.8241, 0.8539, 1.1277, 0.8147, 0.7969, 0.6738]

    # Epoch 1 losses (selected representative batches)
    epoch_1_losses = [0.7760, 0.7289, 0.9707, 0.6988, 0.9668, 0.7986,
                      0.9487, 0.6997, 0.6754, 1.0545, 0.8876, 0.8637, 0.8203, 0.7985]

    # Epoch 2 losses (selected representative batches)
    epoch_2_losses = [0.7498, 0.7005, 0.7011, 0.9004, 0.8000, 0.7615,
                      0.9573, 0.8107, 0.8452, 0.8439, 0.6817, 0.8173, 0.7259, 0.9338]

    # Calculate average loss per epoch
    epoch_avg_losses = [
        np.mean(epoch_0_losses),
        np.mean(epoch_1_losses),
        np.mean(epoch_2_losses)
    ]

    epochs = [0, 1, 2]

    print(f"Average losses per epoch:")
    for i, avg_loss in enumerate(epoch_avg_losses):
        print(f"  Epoch {i}: {avg_loss:.4f}")

    # Create the plot with enhanced styling
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')

    # Define colors - KMPG brand palette (same as improved_process_results.py)
    line_color = '#00338D'  # KMPG blue
    point_color = '#FD349C'  # Pink for data points

    # Plot the training loss line
    line = ax.plot(epochs, epoch_avg_losses,
                   color=line_color, linewidth=4, alpha=0.9,
                   marker='o', markersize=12, markerfacecolor=point_color,
                   markeredgecolor='white', markeredgewidth=3,
                   label='Average Training Loss', zorder=5)

    # Customize the plot with enhanced styling
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('DPO Loss', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.set_title('DPO Training Loss Progression\nQwen3-14B Multi-Model Fine-tuning',
                 fontsize=18, fontweight='bold', pad=25, color='#2C3E50')

    # Set axis properties
    ax.set_xticks(epochs)
    ax.set_xticklabels([f'Epoch {i}' for i in epochs],
                       fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

    # Set proper y-axis scaling (not starting from 0)
    y_min = min(epoch_avg_losses) * 0.95  # 5% padding below minimum
    y_max = max(epoch_avg_losses) * 1.05  # 5% padding above maximum
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.2, 2.2)

    # Add subtle grid (same style as improved_process_results.py)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)

    # Create professional legend
    legend = ax.legend(loc='upper right', fontsize=13, frameon=True,
                       fancybox=True, shadow=True, facecolor='white',
                       edgecolor='gray', framealpha=0.95)
    legend.get_frame().set_linewidth(1.5)

    # Remove top and right spines for cleaner look (same as improved_process_results.py)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#2C3E50')
    ax.spines['bottom'].set_color('#2C3E50')

    # Add statistics text box
    initial_loss = epoch_avg_losses[0]
    final_loss = epoch_avg_losses[-1]
    total_reduction = initial_loss - final_loss
    reduction_pct = (total_reduction / initial_loss) * 100

    stats_text = f'Initial Loss: {initial_loss:.4f}\n'
    stats_text += f'Final Loss: {final_loss:.4f}\n'
    stats_text += f'Total Reduction: {total_reduction:.4f}\n'
    stats_text += f'Improvement: {reduction_pct:.1f}%'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                      alpha=0.8, edgecolor='#00338D'))

    # Add annotations for key improvements
    # Annotate the major drop from epoch 0 to 1
    ax.annotate(f'Major improvement\n-{epoch_avg_losses[0] - epoch_avg_losses[1]:.3f}',
                xy=(0.5, (epoch_avg_losses[0] + epoch_avg_losses[1])/2),
                xytext=(0.7, epoch_avg_losses[0] - 0.05),
                arrowprops=dict(arrowstyle='->', color='#FD349C', lw=2),
                fontsize=11, fontweight='bold', color='#FD349C',
                ha='center')

    # Adjust layout and save
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)

    # Save the plot with high quality
    output_path = 'output_plots/dpo_training_loss_progression.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Training loss plot saved to: {output_path}")

    # Show the plot
    plt.show()

    return fig


def print_loss_analysis():
    """Print detailed analysis of the training loss progression."""
    print("="*60)
    print("DPO TRAINING LOSS ANALYSIS")
    print("="*60)

    # Loss data
    epoch_0_losses = [1.0559, 1.2487, 1.0538, 1.1423, 0.6800, 1.0694,
                      1.0458, 0.9186, 0.8241, 0.8539, 1.1277, 0.8147, 0.7969, 0.6738]
    epoch_1_losses = [0.7760, 0.7289, 0.9707, 0.6988, 0.9668, 0.7986,
                      0.9487, 0.6997, 0.6754, 1.0545, 0.8876, 0.8637, 0.8203, 0.7985]
    epoch_2_losses = [0.7498, 0.7005, 0.7011, 0.9004, 0.8000, 0.7615,
                      0.9573, 0.8107, 0.8452, 0.8439, 0.6817, 0.8173, 0.7259, 0.9338]

    epoch_avgs = [np.mean(losses) for losses in [
        epoch_0_losses, epoch_1_losses, epoch_2_losses]]
    epoch_stds = [np.std(losses) for losses in [
        epoch_0_losses, epoch_1_losses, epoch_2_losses]]

    print(f"\nEpoch-by-Epoch Analysis:")
    for i, (avg, std) in enumerate(zip(epoch_avgs, epoch_stds)):
        print(f"  Epoch {i}:")
        print(f"    Average Loss: {avg:.4f}")
        print(f"    Std Deviation: {std:.4f}")
        print(
            f"    Min Loss: {min([epoch_0_losses, epoch_1_losses, epoch_2_losses][i]):.4f}")
        print(
            f"    Max Loss: {max([epoch_0_losses, epoch_1_losses, epoch_2_losses][i]):.4f}")

        if i > 0:
            improvement = epoch_avgs[i-1] - avg
            improvement_pct = (improvement / epoch_avgs[i-1]) * 100
            print(
                f"    Improvement from Epoch {i-1}: {improvement:.4f} ({improvement_pct:.1f}%)")
        print()

    print(f"Overall Training Summary:")
    total_improvement = epoch_avgs[0] - epoch_avgs[-1]
    total_improvement_pct = (total_improvement / epoch_avgs[0]) * 100
    print(f"  Initial Loss (Epoch 0): {epoch_avgs[0]:.4f}")
    print(f"  Final Loss (Epoch 2): {epoch_avgs[-1]:.4f}")
    print(
        f"  Total Improvement: {total_improvement:.4f} ({total_improvement_pct:.1f}%)")
    print(
        f"  Training Stability: {'Good' if epoch_stds[-1] < epoch_stds[0] else 'Needs monitoring'}")


def main():
    """Main function to create the training loss visualization."""
    print("Creating DPO training loss visualization...")
    print("="*50)

    # Print loss analysis
    print_loss_analysis()

    # Create the plot
    print("\nGenerating training loss plot...")
    fig = create_training_loss_plot()

    print("\nTraining loss visualization complete!")
    print("Check 'output_plots/dpo_training_loss_progression.png' for the result.")


if __name__ == "__main__":
    main()
