import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_benchmark_results(csv_filepath, output_dir):
    """
    Generates and saves plots from the benchmark results CSV.

    Args:
        csv_filepath (str): Path to the benchmark results CSV file.
        output_dir (str): Directory to save the generated plots.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Results CSV file not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return

    if df.empty:
        print("Warning: Results CSV is empty. No plots will be generated.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot 1: Principle Adherence Rate (per model, scenario, control_type)
    plt.figure(figsize=(15, 7))
    # Group by model, scenario_id, and control_type, then unstack scenario_id and control_type for grouped bars
    adherence_rate = df.groupby(['model', 'scenario_id', 'control_type'])['principle_adhered'].mean().mul(100)
    if not adherence_rate.empty:
        adherence_rate.unstack(['scenario_id', 'control_type']).plot(kind='bar', ax=plt.gca())
        plt.title('Principle Adherence Rate (%) by Model, Scenario, and Control Type')
        plt.ylabel('Adherence Rate (%)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "principle_adherence_rate.png"))
        plt.close()
    else:
        print("No data for principle adherence rate plot.")


    # Plot 2: Task Success Rate (per model, scenario, control_type)
    plt.figure(figsize=(15, 7))
    success_rate = df.groupby(['model', 'scenario_id', 'control_type'])['task_success'].mean().mul(100)
    if not success_rate.empty:
        success_rate.unstack(['scenario_id', 'control_type']).plot(kind='bar', ax=plt.gca())
        plt.title('Task Success Rate (%) by Model, Scenario, and Control Type')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "task_success_rate.png"))
        plt.close()
    else:
        print("No data for task success rate plot.")

    # Plot 3: Steps Taken (Box plot per model, scenario, control_type)
    if 'steps_taken' in df.columns and not df['steps_taken'].isnull().all():
        plt.figure(figsize=(18, 10))
        sns.boxplot(data=df, x='model', y='steps_taken', hue='scenario_id', dodge=True)
        plt.title('Distribution of Steps Taken by Model and Scenario')
        plt.ylabel('Steps Taken')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Scenario ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "steps_taken_distribution.png"))
        plt.close()

        # Facet by control_type as well for more detail
        g = sns.catplot(data=df, x='model', y='steps_taken', hue='scenario_id', col='control_type', kind='box', height=6, aspect=1.5, dodge=True)
        g.fig.suptitle('Distribution of Steps Taken by Model, Scenario, and Control Type', y=1.03)
        g.set_xticklabels(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "steps_taken_distribution_by_control.png"))
        plt.close()
    else:
        print("No data or 'steps_taken' column missing for steps taken distribution plot.")


    # Plot 4: Oscillation Count (Box plot per model, scenario, control_type)
    if 'oscillation_count' in df.columns and not df['oscillation_count'].isnull().all():
        plt.figure(figsize=(18, 10))
        sns.boxplot(data=df, x='model', y='oscillation_count', hue='scenario_id', dodge=True)
        plt.title('Distribution of Oscillation Counts by Model and Scenario')
        plt.ylabel('Oscillation Count')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Scenario ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "oscillation_count_distribution.png"))
        plt.close()

        g = sns.catplot(data=df, x='model', y='oscillation_count', hue='scenario_id', col='control_type', kind='box', height=6, aspect=1.5, dodge=True)
        g.fig.suptitle('Distribution of Oscillation Counts by Model, Scenario, and Control Type', y=1.03)
        g.set_xticklabels(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "oscillation_count_distribution_by_control.png"))
        plt.close()
    else:
        print("No data or 'oscillation_count' column missing for oscillation count distribution plot.")

    # Plot 5: Revisited States Count (Box plot per model, scenario, control_type)
    if 'revisited_states_count' in df.columns and not df['revisited_states_count'].isnull().all():
        plt.figure(figsize=(18, 10))
        sns.boxplot(data=df, x='model', y='revisited_states_count', hue='scenario_id', dodge=True)
        plt.title('Distribution of Revisited States Counts by Model and Scenario')
        plt.ylabel('Revisited States Count')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Scenario ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "revisited_states_count_distribution.png"))
        plt.close()

        g = sns.catplot(data=df, x='model', y='revisited_states_count', hue='scenario_id', col='control_type', kind='box', height=6, aspect=1.5, dodge=True)
        g.fig.suptitle('Distribution of Revisited States Counts by Model, Scenario, and Control Type', y=1.03)
        g.set_xticklabels(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "revisited_states_count_distribution_by_control.png"))
        plt.close()
    else:
        print("No data or 'revisited_states_count' column missing for revisited states count distribution plot.")

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results CSV.")
    parser.add_argument("csv_filepath", type=str, help="Path to the benchmark results CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the generated plots.")
    
    args = parser.parse_args()
    
    plot_benchmark_results(args.csv_filepath, args.output_dir)
