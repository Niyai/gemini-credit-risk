# src/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparative_analysis(results_df: pd.DataFrame):
    """
    Creates and saves a bar chart to visualize the comparative analysis results.
    """
    if results_df.empty:
        print("Cannot generate plot because results are empty.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    ax = sns.countplot(
        data=results_df,
        x='Analysis Method',
        hue='Risk Assessment',
        order=['Benchmark ML', 'Baseline LLM', 'Debiased LLM'],
        hue_order=['Low', 'Medium', 'High'],
        palette='magma'
    )
    
    ax.set_title('Comparative Risk Assessment Analysis', fontsize=18, weight='bold')
    ax.set_xlabel('Analysis Method', fontsize=14)
    ax.set_ylabel('Number of Customers', fontsize=14)
    plt.legend(title='Risk Level', loc='upper right')
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('comparative_analysis.png')
    print("\n✅ Comparative analysis plot saved to comparative_analysis.png")


def plot_bias_analysis(results_df: pd.DataFrame):
    """
    Creates and saves a bar chart to visualize the comparative analysis results.
    """
    if results_df.empty:
        print("Cannot generate plot because results are empty.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Define the order for a more logical chart layout
    analysis_order = ['Benchmark ML', 'Baseline LLM', 'Debiased LLM']
    
    ax = sns.countplot(
        data=results_df,
        x='Analysis Method', # <-- CORRECTED: Was 'Analysis Type'
        hue='Risk Assessment',
        order=analysis_order,
        hue_order=['Low', 'Medium', 'High'],
        palette='viridis'
    )
    
    ax.set_title('Comparative Risk Assessment Analysis', fontsize=18, weight='bold')
    ax.set_xlabel('Analysis Method', fontsize=14)
    ax.set_ylabel('Number of Customers', fontsize=14)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Risk Level', loc='upper right')
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('comparative_analysis.png')
    print("\n✅ Comparative analysis plot saved to comparative_analysis.png")


def plot_final_analysis(metrics_df: pd.DataFrame):
    """
    Creates a grouped bar chart to visualize the accuracy vs. fairness trade-off.
    """
    if metrics_df.empty:
        print("Cannot generate plot because metrics are empty.")
        return

    # Melt the DataFrame to prepare it for grouped bar plotting
    df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(
        data=df_melted,
        x='Model',
        y='Score',
        hue='Metric',
        order=['Benchmark ML', 'Baseline LLM', 'Debiased LLM', 'Fine-Tuned LLM'],
        palette='rocket'
    )
    
    ax.set_title('Model Performance: Accuracy vs. Fairness Trade-off', fontsize=18, weight='bold')
    ax.set_xlabel('Model Approach', fontsize=14)
    ax.set_ylabel('Performance Score (0.0 to 1.0)', fontsize=14)
    ax.set_ylim(0, 1) # Set y-axis from 0 to 1 for clarity
    plt.legend(title='Performance Metric', loc='upper right')
    
    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')

    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('accuracy_vs_fairness_tradeoff.png')
    print("\n✅ Final analysis plot saved to accuracy_vs_fairness_tradeoff.png")
