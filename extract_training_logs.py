"""
Extract training data from TensorBoard logs and create visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
from datetime import datetime

def extract_tensorboard_data(log_dir):
    """Extract scalar data from TensorBoard event files"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get all scalar tags
    tags = event_acc.Tags()['scalars']

    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}

    return data

def smooth_curve(values, weight=0.9):
    """Apply exponential moving average smoothing"""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def create_loss_plots(data, output_dir):
    """Create loss curve visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # Define loss groups
    loss_groups = {
        'Generator Losses': ['loss_G', 'loss_G_A2B', 'loss_G_B2A'],
        'Discriminator Losses': ['loss_D', 'loss_D_A', 'loss_D_B'],
        'Cycle Consistency Losses': ['loss_cycle_A', 'loss_cycle_B'],
        'Identity Losses': ['loss_idt_A', 'loss_idt_B'],
    }

    fig_size = (15, 10)

    # Create combined plot
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()

    for idx, (group_name, loss_names) in enumerate(loss_groups.items()):
        ax = axes[idx]

        for loss_name in loss_names:
            if loss_name in data:
                steps = data[loss_name]['steps']
                values = data[loss_name]['values']

                # Plot raw data with transparency
                ax.plot(steps, values, alpha=0.3, label=f'{loss_name} (raw)')

                # Plot smoothed curve
                if len(values) > 10:
                    smoothed = smooth_curve(values)
                    ax.plot(steps, smoothed, linewidth=2, label=f'{loss_name}')

        ax.set_title(group_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('CycleGAN Training Losses - 300 Epochs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_losses_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Create individual plots for each group
    for group_name, loss_names in loss_groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for loss_name in loss_names:
            if loss_name in data:
                steps = data[loss_name]['steps']
                values = data[loss_name]['values']

                # Plot smoothed curve
                if len(values) > 10:
                    smoothed = smooth_curve(values, weight=0.95)
                    ax.plot(steps, smoothed, linewidth=2, label=loss_name)

        ax.set_title(f'{group_name} - 300 Epochs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        filename = group_name.lower().replace(' ', '_') + '.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Loss plots saved to {output_dir}")

def calculate_statistics(data):
    """Calculate statistics for the training data"""
    stats = {}

    for tag, values_dict in data.items():
        values = values_dict['values']
        if len(values) > 0:
            stats[tag] = {
                'initial': values[0],
                'final': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'improvement': values[0] - values[-1] if values[0] > values[-1] else values[-1] - values[0]
            }

    return stats

def main():
    # Log directories
    log_dirs = [
        'experiments/horse2zebra_balanced/logs'
    ]

    output_dir = 'training_report'
    os.makedirs(output_dir, exist_ok=True)

    all_data = {}

    # Extract data from each log directory
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"Processing {log_dir}...")
            event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]

            for event_file in event_files:
                file_path = os.path.join(log_dir, event_file)
                print(f"  Reading {event_file}...")
                try:
                    data = extract_tensorboard_data(file_path)
                    # Merge data
                    for key, value in data.items():
                        if key not in all_data:
                            all_data[key] = value
                        else:
                            # Extend existing data
                            all_data[key]['steps'].extend(value['steps'])
                            all_data[key]['values'].extend(value['values'])
                except Exception as e:
                    print(f"  Error reading {event_file}: {e}")

    if all_data:
        # Create visualizations
        create_loss_plots(all_data, output_dir)

        # Calculate statistics
        stats = calculate_statistics(all_data)

        # Save statistics to JSON
        with open(os.path.join(output_dir, 'training_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"Statistics saved to {output_dir}/training_statistics.json")

        # Print summary
        print("\n=== Training Summary ===")
        for loss_type in ['loss_G', 'loss_D', 'loss_cycle_A', 'loss_cycle_B']:
            if loss_type in stats:
                s = stats[loss_type]
                print(f"\n{loss_type}:")
                print(f"  Initial: {s['initial']:.4f}")
                print(f"  Final: {s['final']:.4f}")
                print(f"  Min: {s['min']:.4f}")
                print(f"  Max: {s['max']:.4f}")
                print(f"  Improvement: {s['improvement']:.4f}")
    else:
        print("No data found in TensorBoard logs")

if __name__ == "__main__":
    main()