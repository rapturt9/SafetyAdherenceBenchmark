import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_benchmark_results(csv_filepath, output_dir):
    df = pd.read_csv(csv_filepath)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Classify scenarios by conflict avoidability (this information might need adjustment)
    # P1-S1 and P1-S2 are conflict-avoidable, P1-S3 and P1-S4 are conflict-unavoidable, etc.
    conflict_avoidable = {
        'P1': ['P1-S1', 'P1-S2'],
        'P2': ['P2-S2', 'P2-S3'],
        'P3': ['P3-S2', 'P3-S4']
    }
    
    # Add conflict type column to the dataframe
    df['conflict_type'] = df.apply(
        lambda row: 'Conflict-Avoidable' if row['scenario_id'] in conflict_avoidable.get(row['principle_id'], []) 
        else 'Conflict-Unavoidable', axis=1
    )

    # 1. Overall Performance Summary (Table)
    overall_summary = df[df['control_type'] == 'Principle_ON'].groupby(['model', 'principle_id'])[['principle_adhered', 'task_success']].mean().reset_index()
    overall_summary.to_csv(os.path.join(output_dir, "overall_performance_summary.csv"), index=False)

    # NEW: Per-model charts showing success rate, principle adherence, and steps taken
    # Group by model to create a separate chart for each
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        # Group by scenario and control type to compute averages
        model_summary = model_df.groupby(['scenario_id', 'control_type']).agg({
            'principle_adhered': 'mean',
            'task_success': 'mean',
            'steps_taken': 'mean'
        }).reset_index()
        
        # Create figure with 3 subplots (one for each metric)
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        
        # Plot 1: Success Rate
        sns.barplot(
            data=model_summary, 
            x='scenario_id', 
            y='task_success', 
            hue='control_type',
            ax=axes[0]
        )
        axes[0].set_title(f'Task Success Rate by Scenario for {model}', fontsize=14)
        axes[0].set_xlabel('Scenario')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title='Control Type')
        
        # Plot 2: Principle Adherence
        sns.barplot(
            data=model_summary,
            x='scenario_id',
            y='principle_adhered',
            hue='control_type',
            ax=axes[1]
        )
        axes[1].set_title(f'Principle Adherence Rate by Scenario for {model}', fontsize=14)
        axes[1].set_xlabel('Scenario')
        axes[1].set_ylabel('Adherence Rate')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title='Control Type')
        
        # Plot 3: Average Steps Taken
        sns.barplot(
            data=model_summary,
            x='scenario_id',
            y='steps_taken',
            hue='control_type',
            ax=axes[2]
        )
        axes[2].set_title(f'Average Steps Taken by Scenario for {model}', fontsize=14)
        axes[2].set_xlabel('Scenario')
        axes[2].set_ylabel('Steps Taken')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].legend(title='Control Type')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model.split('/')[-1]}_scenario_metrics.png"))
        plt.close()

    # 2. Principle Adherence Rate by Conflict Type (Bar Chart)
    adherence_df = df.groupby(['model', 'principle_id', 'conflict_type', 'control_type'])['principle_adhered'].mean().reset_index()
    for principle in adherence_df['principle_id'].unique():
        principle_data = adherence_df[
            (adherence_df['principle_id'] == principle) & 
            (adherence_df['control_type'] == 'Principle_ON')
        ]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=principle_data, x='model', y='principle_adhered', hue='conflict_type')
        plt.title(f'Principle Adherence Rate by Conflict Type for {principle}')
        plt.ylabel('Adherence Rate')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Conflict Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"adherence_rate_{principle}.png"))
        plt.close()

    # 3. Task Success Rate by Conflict Type (Bar Chart)
    success_df = df.groupby(['model', 'principle_id', 'conflict_type', 'control_type'])['task_success'].mean().reset_index()
    for principle in success_df['principle_id'].unique():
        principle_data = success_df[success_df['principle_id'] == principle]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=principle_data, x='model', y='task_success', hue='control_type')
        plt.title(f'Task Success Rate for {principle}')
        plt.ylabel('Success Rate')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Control Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"success_rate_{principle}.png"))
        plt.close()

    # 4. Impact of Principle ON vs. OFF on Specific Violations (Bar Chart)
    violation_df = df.groupby(['model', 'principle_id', 'control_type'])['principle_adhered'].mean().reset_index()
    for principle in violation_df['principle_id'].unique():
        principle_data = violation_df[violation_df['principle_id'] == principle]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=principle_data, x='model', y='principle_adhered', hue='control_type')
        plt.title(f'Impact of Principle ON vs OFF for {principle}')
        plt.ylabel('Adherence Rate')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Control Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"impact_principle_{principle}.png"))
        plt.close()

    # 5. Behavioral Metrics - Steps Taken (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model', y='steps_taken', hue='control_type')
    plt.title('Steps Taken by Model and Control Type')
    plt.ylabel('Steps Taken')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Control Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "steps_taken.png"))
    plt.close()

    # 6. Behavioral Metrics - Oscillation/Revisit Counts (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model', y='oscillation_count', hue='control_type')
    plt.title('Oscillation Counts by Model and Control Type')
    plt.ylabel('Oscillation Count')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Control Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "oscillation_counts.png"))
    plt.close()

    # 7. Scatter Plot: PAR vs. TSR
    scatter_df = df[df['control_type'] == 'Principle_ON'].groupby(['model', 'principle_id'])[['principle_adhered', 'task_success']].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=scatter_df, x='principle_adhered', y='task_success', hue='principle_id', style='model', s=100)
    plt.title('PAR vs TSR (Principle ON)')
    plt.xlabel('Principle Adherence Rate (PAR)')
    plt.ylabel('Task Success Rate (TSR)')
    plt.legend(title='Principle ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "par_vs_tsr.png"))
    plt.close()

    # 8. Heatmap of Principle Violations by Scenario Type
    heatmap_df = df.pivot_table(index='model', columns='scenario_id', values='principle_adhered', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Adherence Rate'})
    plt.title('Principle Violations by Scenario Type')
    plt.ylabel('Model')
    plt.xlabel('Scenario ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "violations_heatmap.png"))
    plt.close()

    # 9. Stacked Bar Chart for P3 Procedural Violations
    p3_df = df[df['principle_id'] == 'P3']
    p3_summary = p3_df.groupby(['model', 'control_type'])['steps_taken'].sum().unstack()
    p3_summary.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('P3 Procedural Violations')
    plt.ylabel('Total Steps Taken')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Control Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p3_procedural_violations.png"))
    plt.close()

    # 10. "Frustration" Index (Composite Bar Chart)
    df['frustration_index'] = (df['steps_taken'] + df['oscillation_count'] + df['revisited_states_count']) / 3
    frustration_df = df.groupby(['model', 'principle_id'])['frustration_index'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=frustration_df, x='model', y='frustration_index', hue='principle_id')
    plt.title('Frustration Index by Model and Principle')
    plt.ylabel('Frustration Index')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Principle ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frustration_index.png"))
    plt.close()

    # 11. Detailed Task Success Breakdown (Bar Chart)
    for principle in df['principle_id'].unique():
        principle_data = df[df['principle_id'] == principle]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=principle_data, x='scenario_id', y='task_success', hue='control_type')
        plt.title(f'Task Success Breakdown for {principle}')
        plt.ylabel('Task Success Rate')
        plt.xlabel('Scenario ID')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Control Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"task_success_breakdown_{principle}.png"))
        plt.close()
        
    # NEW: 12. Full Picture Scatter Plot (Success Rate vs Principle Adherence)
    # Group by all the dimensions we care about
    full_picture_df = df.groupby(['model', 'principle_id', 'conflict_type', 'control_type'])[
        ['principle_adhered', 'task_success']
    ].mean().reset_index()
    
    # Create a combined category for plotting
    full_picture_df['category'] = full_picture_df.apply(
        lambda row: f"{row['principle_id']}-{row['conflict_type']}-{row['control_type']}", axis=1
    )
    
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=full_picture_df, 
        x='principle_adhered', 
        y='task_success', 
        hue='category',
        style='model',
        s=150,
        alpha=0.8
    )
    
    # Improve the legend handling for many items
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Success Rate vs. Principle Adherence - Full Picture')
    plt.xlabel('Principle Adherence Rate')
    plt.ylabel('Task Success Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_picture_scatter.png"))
    plt.close()
    
    # NEW: 13. Comprehensive Grid View of All Conditions
    # Create a facet grid to show all combinations
    principle_ids = df['principle_id'].unique()
    
    for principle_id in principle_ids:
        filtered_df = full_picture_df[full_picture_df['principle_id'] == principle_id]
        
        g = sns.catplot(
            data=filtered_df,
            x='conflict_type',
            y='task_success',
            hue='control_type',
            col='model',
            kind='bar',
            height=5,
            aspect=0.8,
            palette='Set2',
            legend=False
        )
        
        g.fig.suptitle(f'Task Success by Conflict Type and Control Type for {principle_id}', y=1.05)
        g.set_axis_labels("Conflict Type", "Task Success Rate")
        g.set_titles("Model: {col_name}")
        plt.legend(title='Control Type', loc='upper right', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comprehensive_grid_{principle_id}.png"))
        plt.close()
    
    # NEW: 14. Adherence vs Success Facet Grid by Principle, Control Type, and Conflict Type
    fg = sns.FacetGrid(
        data=full_picture_df,
        col='principle_id',
        row='control_type',
        hue='conflict_type',
        height=4,
        aspect=1.2
    )
    
    fg.map(sns.scatterplot, "principle_adhered", "task_success", s=100, alpha=0.8)
    fg.add_legend(title="Conflict Type", loc='upper right')
    fg.set_axis_labels("Principle Adherence Rate", "Task Success Rate")
    fg.set_titles(col_template="Principle: {col_name}", row_template="Control: {row_name}")
    plt.savefig(os.path.join(output_dir, "adherence_vs_success_grid.png"))
    plt.close()

    # NEW: 15. Model Performance Scatter (Principle ON)
    # Filter for Principle_ON control type and Conflict-Avoidable scenarios
    principle_on_avoidable_df = df[
        (df['control_type'] == 'Principle_ON') & 
        (df['conflict_type'] == 'Conflict-Avoidable')
    ]
    
    # Group by model and calculate mean adherence and success
    model_performance_df = principle_on_avoidable_df.groupby('model')[['principle_adhered', 'task_success']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=model_performance_df,
        x='principle_adhered',
        y='task_success',
        hue='model',
        s=150,  # Adjust size of points
        alpha=0.8
    )
    
    plt.title('Model Performance: Adherence vs. Success (Principle ON, Conflict-Avoidable)')
    plt.xlabel('Average Principle Adherence Rate')
    plt.ylabel('Average Task Success Rate')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_performance_scatter_principle_on_conflict_avoidable.png"))
    plt.close()

    print(f"All plots saved to {output_dir}")

if __name__ == "__main__": 


    csv_filepath = os.path.join(os.path.dirname(__file__), "benchmark_results_all.csv") # define here
    output_dir = os.path.join(os.path.dirname(__file__), "benchmark_plots") # define here
    plot_benchmark_results(csv_filepath, output_dir)
