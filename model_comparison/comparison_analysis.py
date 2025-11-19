"""
Comprehensive Model Comparison Analysis
Wine Quality Prediction - Supervised vs Unsupervised Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
OUTPUT_DIR = "/home/user/5126-Team-Project/model_comparison/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def create_model_performance_comparison():
    """
    Create comprehensive performance comparison table and visualization
    """
    print("Creating model performance comparison...")

    # Supervised model results (from code analysis)
    supervised_models = pd.DataFrame({
        'Model': ['Decision Tree', 'XGBoost', 'Random Forest', 'Deep Neural Network'],
        'Accuracy': [0.7867, 0.8787, 0.8695, 0.7575],
        'Precision': [np.nan, np.nan, np.nan, 0.799],
        'Recall': [np.nan, np.nan, np.nan, 0.818],
        'F1-Score': [0.448, 0.554, 0.471, 0.809],
        'G-mean': [0.580, 0.688, 0.610, np.nan],
        'Type': ['Tree-based', 'Tree-based', 'Tree-based', 'Neural Network'],
        'Interpretability': ['High', 'Medium', 'Medium', 'Low'],
        'Training Time': ['Fast', 'Medium', 'Medium', 'Slow'],
        'Status': ['Deleted', 'Deleted', 'Deleted', 'Active']
    })

    # Unsupervised clustering results
    unsupervised_models = pd.DataFrame({
        'Model': ['K-Means (Red Wine)', 'K-Means (White Wine)'],
        'Optimal K': [2, 2],
        'Silhouette Score': [0.205, 0.213],
        'Type': ['Clustering', 'Clustering'],
        'Purpose': ['Pattern Discovery', 'Pattern Discovery']
    })

    # Save tables
    supervised_models.to_csv(os.path.join(OUTPUT_DIR, 'supervised_models_comparison.csv'), index=False)
    unsupervised_models.to_csv(os.path.join(OUTPUT_DIR, 'unsupervised_models_comparison.csv'), index=False)

    # Visualization 1: Supervised Model Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d62728' if status == 'Deleted' else '#2ca02c'
              for status in supervised_models['Status']]
    bars = ax.barh(supervised_models['Model'], supervised_models['Accuracy'], color=colors, alpha=0.7)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, supervised_models['Accuracy'])):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.2%}', va='center', fontweight='bold')

    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Supervised Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')

    # Legend for status
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ca02c', alpha=0.7, label='Active'),
                      Patch(facecolor='#d62728', alpha=0.7, label='Deleted (Previous)')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_accuracy_comparison.png")

    # Visualization 2: F1-Score Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(supervised_models['Model'], supervised_models['F1-Score'],
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)

    for i, (model, f1) in enumerate(zip(supervised_models['Model'], supervised_models['F1-Score'])):
        if not np.isnan(f1):
            ax.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontweight='bold')

    ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Supervised Model F1-Score Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_f1_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_f1_comparison.png")

    # Visualization 3: Multi-metric radar chart for active model (DNN)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    dnn_values = [0.7575, 0.799, 0.818, 0.809]

    # Normalize to 0-1 scale
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    dnn_values_plot = dnn_values + [dnn_values[0]]  # Complete the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, dnn_values_plot, 'o-', linewidth=2, label='DNN', color='#d62728')
    ax.fill(angles, dnn_values_plot, alpha=0.25, color='#d62728')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Deep Neural Network - Performance Metrics',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dnn_metrics_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: dnn_metrics_radar.png")

    # Visualization 4: Clustering quality distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Red wine clustering
    red_cluster_quality = np.array([
        [3, 11, 180, 224, 114, 10],  # Cluster 1
        [7, 42, 397, 311, 53, 7]      # Cluster 2
    ])
    qualities_red = [3, 4, 5, 6, 7, 8]

    x = np.arange(len(qualities_red))
    width = 0.35
    ax1.bar(x - width/2, red_cluster_quality[0], width, label='Cluster 1', alpha=0.8, color='#1f77b4')
    ax1.bar(x + width/2, red_cluster_quality[1], width, label='Cluster 2', alpha=0.8, color='#ff7f0e')
    ax1.set_xlabel('Quality Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax1.set_title('Red Wine - Cluster Distribution by Quality', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(qualities_red)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # White wine clustering
    white_cluster_quality = np.array([
        [8, 104, 515, 1134, 600, 119, 4],    # Cluster 1
        [12, 49, 660, 654, 89, 12, 1]         # Cluster 2
    ])
    qualities_white = [3, 4, 5, 6, 7, 8, 9]

    x = np.arange(len(qualities_white))
    ax2.bar(x - width/2, white_cluster_quality[0], width, label='Cluster 1', alpha=0.8, color='#2ca02c')
    ax2.bar(x + width/2, white_cluster_quality[1], width, label='Cluster 2', alpha=0.8, color='#d62728')
    ax2.set_xlabel('Quality Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax2.set_title('White Wine - Cluster Distribution by Quality', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(qualities_white)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_quality_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: clustering_quality_distribution.png")

    return supervised_models, unsupervised_models


def create_performance_summary():
    """
    Create a comprehensive performance summary table
    """
    print("\nCreating performance summary...")

    summary_data = {
        'Approach': ['Supervised Learning', 'Supervised Learning', 'Supervised Learning',
                     'Supervised Learning', 'Unsupervised Learning'],
        'Model': ['XGBoost (Best)', 'Random Forest', 'Decision Tree',
                  'Deep Neural Network (Current)', 'K-Means Clustering'],
        'Key Metric': ['87.87% Accuracy', '86.95% Accuracy', '78.67% Accuracy',
                       '75.75% Accuracy, 80.9% F1', 'Silhouette: 0.205-0.213'],
        'Strengths': [
            'Highest accuracy, good F1-score, handles imbalanced data',
            'Good accuracy, robust to overfitting, feature importance',
            'Highly interpretable, fast training, simple rules',
            'Good recall (81.8%), captures complex patterns, high F1',
            'Reveals natural groupings, no labels needed, pattern discovery'
        ],
        'Weaknesses': [
            'Less interpretable, requires tuning, code deleted',
            'Less interpretable, slower than single tree, code deleted',
            'Lowest accuracy, prone to overfitting, code deleted',
            'Lowest accuracy, slow training, requires many epochs, less interpretable',
            'No direct quality prediction, moderate silhouette scores, needs interpretation'
        ],
        'Best Use Case': [
            'Production deployment (if restored)',
            'Feature importance analysis (if restored)',
            'Quick baseline, educational purposes',
            'Current production model, complex pattern detection',
            'Exploratory analysis, customer segmentation'
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.csv'), index=False)

    print("\nPerformance Summary:")
    print("=" * 100)
    for idx, row in summary_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  Key Metric: {row['Key Metric']}")
        print(f"  Strengths: {row['Strengths']}")
        print(f"  Weaknesses: {row['Weaknesses']}")
        print(f"  Best Use: {row['Best Use Case']}")
    print("=" * 100)

    return summary_df


def create_insights_summary():
    """
    Generate key insights from model comparison
    """
    print("\nGenerating key insights...")

    insights = {
        'Category': [
            'Best Performing Model',
            'Most Balanced Model',
            'Most Interpretable',
            'Current Model Status',
            'Clustering Insights',
            'Data Quality',
            'Feature Engineering',
            'Class Imbalance'
        ],
        'Insight': [
            'XGBoost achieved the highest accuracy (87.87%) but was deleted from the repository',
            'Deep Neural Network shows the best balance of precision (79.9%) and recall (81.8%) with F1=0.809',
            'Decision Tree offers highest interpretability but lowest accuracy (78.67%)',
            'Only DNN is currently active; all tree-based models were removed, limiting model comparison options',
            'K-Means clustering reveals 2 natural groups in both red and white wines with moderate silhouette scores (0.205-0.213)',
            'Clusters show correlation with quality: Cluster 1 tends toward higher quality wines',
            'Engineered features (acidity_ratio, alcohol_va_ratio, etc.) improved model performance',
            'All models struggle with minority class (high-quality wines), indicating class imbalance'
        ],
        'Impact': [
            'Critical',
            'High',
            'Medium',
            'High',
            'Medium',
            'High',
            'High',
            'Critical'
        ],
        'Recommendation': [
            'Consider restoring XGBoost or retraining for production use',
            'DNN is suitable for current deployment but consider ensemble methods',
            'Keep a simple tree model for stakeholder explanations',
            'Restore deleted models for comprehensive comparison and ensemble options',
            'Use clustering for customer segmentation and targeted marketing',
            'Validate clustering patterns with domain experts in winemaking',
            'Continue feature engineering with domain knowledge',
            'Implement SMOTE, class weights, or stratified sampling to address imbalance'
        ]
    }

    insights_df = pd.DataFrame(insights)
    insights_df.to_csv(os.path.join(OUTPUT_DIR, 'key_insights.csv'), index=False)

    # Create insights visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color code by impact
    colors_map = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#2ca02c'}
    colors = [colors_map[impact] for impact in insights_df['Impact']]

    y_pos = np.arange(len(insights_df))
    ax.barh(y_pos, [1]*len(insights_df), color=colors, alpha=0.3)

    # Add text
    for i, (cat, insight) in enumerate(zip(insights_df['Category'], insights_df['Insight'])):
        ax.text(0.02, i, f"{cat}:\n{insight[:80]}...",
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim([0, 1])
    ax.set_xticks([])
    ax.set_title('Key Insights from Model Comparison', fontsize=14, fontweight='bold', pad=20)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.3, label='Critical Impact'),
        Patch(facecolor='#ff7f0e', alpha=0.3, label='High Impact'),
        Patch(facecolor='#2ca02c', alpha=0.3, label='Medium Impact')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'key_insights_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: key_insights_summary.png")

    return insights_df


def create_model_trade_offs():
    """
    Create visualization of model trade-offs
    """
    print("\nCreating model trade-offs analysis...")

    # Model characteristics (0-10 scale)
    models = ['Decision Tree', 'XGBoost', 'Random Forest', 'DNN']
    characteristics = {
        'Accuracy': [7.9, 8.8, 8.7, 7.6],
        'Interpretability': [9.0, 5.0, 6.0, 3.0],
        'Training Speed': [9.0, 6.0, 5.0, 3.0],
        'Prediction Speed': [9.0, 7.0, 6.0, 8.0],
        'Scalability': [6.0, 8.0, 7.0, 9.0]
    }

    # Create radar chart for each model
    categories = list(characteristics.keys())
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (model, color) in enumerate(zip(models, colors)):
        values = [characteristics[cat][idx] for cat in categories]
        values += values[:1]

        ax = axes[idx]
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 10)
        ax.set_title(model, fontsize=12, fontweight='bold', pad=20, color=color)
        ax.grid(True)

    plt.suptitle('Model Trade-offs Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_tradeoffs.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_tradeoffs.png")


if __name__ == "__main__":
    print("="*80)
    print("WINE QUALITY PREDICTION - COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("="*80)

    # Create all analyses
    supervised_df, unsupervised_df = create_model_performance_comparison()
    summary_df = create_performance_summary()
    insights_df = create_insights_summary()
    create_model_trade_offs()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*80)

    print("\nGenerated Files:")
    print("  1. supervised_models_comparison.csv")
    print("  2. unsupervised_models_comparison.csv")
    print("  3. model_performance_summary.csv")
    print("  4. key_insights.csv")
    print("  5. model_accuracy_comparison.png")
    print("  6. model_f1_comparison.png")
    print("  7. dnn_metrics_radar.png")
    print("  8. clustering_quality_distribution.png")
    print("  9. key_insights_summary.png")
    print(" 10. model_tradeoffs.png")
