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


def perform_exploratory_data_analysis(df: pd.DataFrame):
    """
    Generates and saves a series of plots for Exploratory Data Analysis (EDA)
    to understand the relationships and biases within the dataset itself.
    """
    print("\n--- 🔬 Performing Exploratory Data Analysis (EDA) ---")
    
    # Set the visual style for the plots
    sns.set_theme(style="whitegrid")

    # 1. Delinquency Rate by Gender
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='gender', y='isdelinquent', palette='pastel')
    plt.title('Delinquency Rate by Gender', fontsize=16)
    plt.ylabel('Delinquency Rate')
    plt.savefig('eda_delinquency_by_gender.png')
    print("✅ EDA plot saved: delinquency_by_gender.png")

    # 2. Delinquency Rate by Primary State (Top 10 states)
    plt.figure(figsize=(12, 7))
    top_states = df['primary_state'].value_counts().nlargest(10).index
    sns.barplot(data=df[df['primary_state'].isin(top_states)], x='primary_state', y='isdelinquent', palette='muted')
    plt.title('Delinquency Rate by Top 10 States', fontsize=16)
    plt.ylabel('Delinquency Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda_delinquency_by_state.png')
    print("✅ EDA plot saved: delinquency_by_state.png")

    # 3. Age Distribution for Delinquent vs. Non-Delinquent Customers
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='isdelinquent', multiple='stack', palette='rocket', bins=20)
    plt.title('Age Distribution by Delinquency Status', fontsize=16)
    plt.savefig('eda_age_distribution.png')
    print("✅ EDA plot saved: age_distribution.png")


def plot_final_analysis(metrics_df: pd.DataFrame):
    """
    Creates a grouped bar chart to visualize the accuracy vs. fairness trade-off.
    """
    if metrics_df.empty:
        print("Cannot generate final plot because metrics are empty.")
        return

    # Melt the DataFrame to prepare it for grouped bar plotting
    df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    
    ax = sns.barplot(
        data=df_melted,
        x='Model',
        y='Score',
        hue='Metric',
        order=['Benchmark ML', 'Baseline LLM', 'Debiased LLM', 'Fine-Tuned LLM'],
        palette='viridis'
    )
    
    ax.set_title('Model Performance: Accuracy vs. Fairness Trade-off', fontsize=18, weight='bold')
    ax.set_xlabel('Model Approach', fontsize=14)
    ax.set_ylabel('Performance Score (0.0 to 1.0)', fontsize=14)
    ax.set_ylim(0, 1)
    plt.legend(title='Performance Metric', loc='upper right')
    
    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')

    plt.tight_layout()
    
    plt.savefig('final_accuracy_vs_fairness.png')
    print("\n✅ Final analysis plot saved to final_accuracy_vs_fairness.png")
