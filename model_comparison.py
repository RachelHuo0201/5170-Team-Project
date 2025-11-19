import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================
# MODEL METRICS 
# ============================================

# DNN metrics 
dnn_metrics = {
    'Accuracy': 75.76,
    'Precision': 79.91,
    'Recall': 81.83,
    'F1-Score': 80.86
}

# XGBoost metrics from cv_summary
xgboost_metrics = {
    'Accuracy': 75.69,
    'Precision': 79.07,
    'Recall': 83.25,
    'F1-Score': 81.08
}

# ============================================
# CREATE COMPARISON DATAFRAME
# ============================================

metrics_df = pd.DataFrame({
    'Metric': list(dnn_metrics.keys()),
    'DNN': list(dnn_metrics.values()),
    'XGBoost': list(xgboost_metrics.values())
})

# Calculate differences
metrics_df['Difference'] = metrics_df['XGBoost'] - metrics_df['DNN']

print("=" * 70)
print("MODEL COMPARISON: XGBoost vs Deep Neural Network")
print("=" * 70)
print(metrics_df.to_string(index=False))
print("\n")

# ============================================
# CREATE BAR CHART
# ============================================

fig, ax = plt.subplots(figsize=(12, 7))

# Set positions for bars
x = np.arange(len(metrics_df['Metric']))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, metrics_df['DNN'], width, 
               label='DNN', color='#f5576c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, metrics_df['XGBoost'], width, 
               label='XGBoost', color='#43e97b', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Customize chart
ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('XGBoost vs DNN Performance Comparison\nWine Quality Prediction', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Metric'], fontsize=12)
ax.set_ylim(0, 100)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add horizontal line at 80% for reference
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(len(x)-0.5, 81, '80% threshold', fontsize=9, color='gray', style='italic')

plt.tight_layout()
plt.savefig('xgboost_vs_dnn_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Chart saved as 'xgboost_vs_dnn_comparison.png'")
plt.show()

# ============================================
# DETAILED MODEL COMPARISON
# ============================================

print("\n" + "=" * 70)
print("DETAILED MODEL ANALYSIS")
print("=" * 70)

print("\n🧠 DEEP NEURAL NETWORK (DNN)")
print("-" * 70)
print(f"  Accuracy:  {dnn_metrics['Accuracy']:.2f}%")
print(f"  Precision: {dnn_metrics['Precision']:.2f}%")
print(f"  Recall:    {dnn_metrics['Recall']:.2f}%")
print(f"  F1-Score:  {dnn_metrics['F1-Score']:.2f}%")
print("\n  Architecture:")
print("    • Input: 14 physicochemical features")
print("    • Hidden Layers: 3 layers × 128 neurons")
print("    • Regularization: BatchNorm + Dropout(0.2)")
print("    • Training: 5000 epochs, early stop at 300")
print("    • Optimizer: Adam (lr=1e-5, weight_decay=5e-4)")
print("\n  Confusion Matrix:")
print("    TN: 261  |  FP: 137")
print("    FN: 121  |  TP: 545")

print("\n\n🚀 XGBoost (Gradient Boosting)")
print("-" * 70)
print(f"  Accuracy:  {xgboost_metrics['Accuracy']:.2f}%")
print(f"  Precision: {xgboost_metrics['Precision']:.2f}%")
print(f"  Recall:    {xgboost_metrics['Recall']:.2f}% ⭐ (Highest)")
print(f"  F1-Score:  {xgboost_metrics['F1-Score']:.2f}%")
print("\n  Strengths:")
print("    • Superior lift curve performance")
print("    • Highest recall (83.25%) - best at identifying quality wines")
print("    • Better for tabular/chemical data")
print("    • Feature importance interpretability")
print("    • Faster inference time")
print("    • ✓ Selected as FINAL MODEL")

# ============================================
# KEY INSIGHTS
# ============================================

print("\n\n" + "=" * 70)
print("📊 KEY INSIGHTS")
print("=" * 70)

print("\n1. PERFORMANCE COMPARISON:")
print(f"   • Very similar accuracy: XGBoost {xgboost_metrics['Accuracy']:.2f}% vs DNN {dnn_metrics['Accuracy']:.2f}%")
print(f"   • XGBoost wins on Recall: {xgboost_metrics['Recall']:.2f}% vs {dnn_metrics['Recall']:.2f}% (+{xgboost_metrics['Recall']-dnn_metrics['Recall']:.2f}%)")
print(f"   • DNN wins on Precision: {dnn_metrics['Precision']:.2f}% vs {xgboost_metrics['Precision']:.2f}% (+{dnn_metrics['Precision']-xgboost_metrics['Precision']:.2f}%)")
print(f"   • Similar F1-Score: XGBoost {xgboost_metrics['F1-Score']:.2f}% vs DNN {dnn_metrics['F1-Score']:.2f}%")

print("\n2. MODEL SELECTION RATIONALE:")
print("   ✓ XGBoost's higher recall (83.25%) is crucial for wine quality")
print("   ✓ Missing a quality wine is worse than false positives")
print("   ✓ Superior lift curve = better ranking capability")
print("   ✓ Feature importance aids winemaker decision-making")

print("\n3. BOTH MODELS VALIDATE:")
print("   • Wine quality CAN be predicted from chemical properties")
print("   • ~75-76% accuracy is achievable")
print("   • F1-scores >80% indicate robust classification")

# ============================================
# MODEL SELECTION JUSTIFICATION
# ============================================

print("\n\n" + "=" * 70)
print("MODEL SELECTION: Why XGBoost Over DNN?")
print("=" * 70)
print("""
1. RECALL ADVANTAGE (+1.42%)
   - XGBoost: 83.25% vs DNN: 81.83%
   - Better at identifying high-quality wines
   - Fewer false negatives = fewer missed quality wines

2. LIFT CURVE SUPERIORITY
   - XGBoost showed the best lift curve performance
   - Critical for ranking predictions in quality assessment
   - Better separates quality tiers

3. TABULAR DATA OPTIMIZATION
   - Physicochemical features are inherently tabular
   - XGBoost excels at capturing feature interactions
   - Tree-based methods natural fit for chemical data

4. INTERPRETABILITY
   - Feature importance provides actionable insights
   - Winemakers can understand decision drivers
   - Clear path to production improvements

5. PRODUCTION EFFICIENCY
   - Faster inference than deep neural networks
   - More stable predictions across datasets
   - Easier to deploy and maintain
   - Lower computational requirements
""")


# ============================================
# SUMMARY STATISTICS
# ============================================

print("\n" + "=" * 70)
print("📈 SUMMARY STATISTICS")
print("=" * 70)

print("\nPERFORMANCE METRICS:")
for metric in metrics_df['Metric']:
    dnn_val = metrics_df[metrics_df['Metric']==metric]['DNN'].values[0]
    xgb_val = metrics_df[metrics_df['Metric']==metric]['XGBoost'].values[0]
    diff = xgb_val - dnn_val
    winner = "XGBoost" if diff > 0 else "DNN" if diff < 0 else "Tie"
    
    print(f"\n{metric}:")
    print(f"  DNN:     {dnn_val:.2f}%")
    print(f"  XGBoost: {xgb_val:.2f}%")
    print(f"  Diff:    {diff:+.2f}% → Winner: {winner}")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE")
print("=" * 70)
