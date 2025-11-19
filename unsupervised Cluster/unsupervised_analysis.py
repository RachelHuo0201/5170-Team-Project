"""
Unsupervised K-Means Clustering & Visualization
Python implementation for wine quality analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import os

# Configuration
PROJECT_ROOT = "/home/user/5126-Team-Project/unsupervised Cluster"
OUT_ROOT = os.path.join(PROJECT_ROOT, "out")
SEED = 42
np.random.seed(SEED)

# Create output directory
os.makedirs(OUT_ROOT, exist_ok=True)

def run_unsupervised_clustering(cleaned_csv, prefix, out_dir, seed=42):
    """
    Run K-Means clustering with automatic k selection using silhouette analysis

    Args:
        cleaned_csv: Path to cleaned dataset
        prefix: Prefix for output files (e.g., 'red' or 'white')
        out_dir: Output directory for results
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"Running unsupervised clustering for: {prefix}")
    print(f"{'='*60}")

    # Load data
    data_clean = pd.read_csv(cleaned_csv)
    print(f"Loaded data: {data_clean.shape}")

    # Select numeric features (exclude quality)
    num_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [col for col in num_cols if col != 'quality']

    X = data_clean[num_cols].copy()
    print(f"Features used: {num_cols}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal k using silhouette analysis
    k_range = range(2, 9)
    silhouette_scores = []

    print("\nTesting k values from 2 to 8...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=25, max_iter=100, random_state=seed)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"  k={k}: silhouette score = {silhouette_avg:.4f}")

    # Find best k
    best_k_idx = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_k_idx]
    best_score = silhouette_scores[best_k_idx]
    print(f"\n[{prefix}] Best k = {best_k} (mean silhouette = {best_score:.3f})")

    # Fit final model with best k
    kmeans_best = KMeans(n_clusters=best_k, n_init=25, max_iter=200, random_state=seed)
    clusters = kmeans_best.fit_predict(X_scaled)

    # Save silhouette plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o', linewidth=2)
    plt.xlabel('Number of clusters (k)', fontsize=11)
    plt.ylabel('Mean silhouette width', fontsize=11)
    plt.title(f'{prefix} - Mean Silhouette (k selection)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_silhouette.png'), dpi=300)
    plt.close()
    print(f"Saved: {prefix}_silhouette.png")

    # PCA visualization
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis',
                         alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11)
    plt.title(f'{prefix} - PCA of Final Clusters (k = {best_k})', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_cluster_pca.png'), dpi=300)
    plt.close()
    print(f"Saved: {prefix}_cluster_pca.png")

    # Cluster centers heatmap
    centers_scaled = kmeans_best.cluster_centers_
    centers_df = pd.DataFrame(centers_scaled, columns=num_cols)
    centers_df.index = [f'Cluster {i+1}' for i in range(best_k)]

    plt.figure(figsize=(12, 6))
    sns.heatmap(centers_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean (z-score)'}, linewidths=0.5)
    plt.title(f'{prefix} - Cluster Centers (z-scored Feature Means)', fontsize=12, fontweight='bold')
    plt.xlabel('Features', fontsize=11)
    plt.ylabel('Cluster', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_centers_heatmap.png'), dpi=300)
    plt.close()
    print(f"Saved: {prefix}_centers_heatmap.png")

    # Cross-tabulation: cluster vs quality
    data_clean['cluster'] = clusters + 1  # 1-indexed clusters
    cross_tab = pd.crosstab(data_clean['cluster'], data_clean['quality'])
    cross_tab.to_csv(os.path.join(out_dir, f'{prefix}_cluster_vs_quality_numeric.csv'))

    # Percentage cross-tab (column-wise percentages)
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=0), axis=1) * 100
    cross_tab_pct.to_csv(os.path.join(out_dir, f'{prefix}_cluster_vs_quality_percent.csv'))

    # Heatmap of cluster vs quality
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Proportion (%)'}, linewidths=0.5,
                vmin=0, vmax=100)
    plt.title(f'{prefix} - Cluster vs Quality (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Quality (1-9)', fontsize=11)
    plt.ylabel('Cluster', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_cluster_quality_heatmap_percent.png'), dpi=300)
    plt.close()
    print(f"Saved: {prefix}_cluster_quality_heatmap_percent.png")

    # Print cluster statistics
    print(f"\nCluster Statistics for {prefix}:")
    print(f"{'Cluster':<10} {'Size':<10} {'% of Data':<12}")
    print("-" * 32)
    for i in range(best_k):
        cluster_size = np.sum(clusters == i)
        cluster_pct = 100 * cluster_size / len(clusters)
        print(f"{i+1:<10} {cluster_size:<10} {cluster_pct:>6.2f}%")

    print(f"\nCross-tabulation (Cluster vs Quality):")
    print(cross_tab)

    return {
        'best_k': best_k,
        'silhouette_score': best_score,
        'clusters': clusters,
        'centers': centers_df,
        'cross_tab': cross_tab,
        'cross_tab_pct': cross_tab_pct
    }


def plot_quality_centers_heatmap(cleaned_csv, prefix, out_dir, use_zscore=True):
    """
    Create heatmap of feature means grouped by quality scores

    Args:
        cleaned_csv: Path to cleaned dataset
        prefix: Prefix for output files
        out_dir: Output directory
        use_zscore: Whether to use z-scored features
    """
    print(f"\n{'='*60}")
    print(f"Creating quality centers heatmap for: {prefix}")
    print(f"{'='*60}")

    df = pd.read_csv(cleaned_csv)

    # Select numeric features (exclude quality)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [col for col in num_cols if col != 'quality']

    X = df[num_cols].copy()

    # Standardize if requested
    if use_zscore:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=num_cols)
    else:
        X_scaled = X

    # Group by quality and compute means
    X_scaled['quality'] = df['quality']
    quality_centers = X_scaled.groupby('quality').mean()

    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(quality_centers.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean (z-score)'}, linewidths=0.5)
    plt.title(f'{prefix} - Quality Group Centers (z-scored Feature Means)', fontsize=12, fontweight='bold')
    plt.xlabel('Quality', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_quality_centers_quality_z.png'), dpi=300)
    plt.close()
    print(f"Saved: {prefix}_quality_centers_quality_z.png")

    return quality_centers


if __name__ == "__main__":
    # Run unsupervised clustering for red wine
    red_results = run_unsupervised_clustering(
        cleaned_csv=os.path.join(PROJECT_ROOT, "red_cleaned.csv"),
        prefix="red",
        out_dir=OUT_ROOT,
        seed=SEED
    )

    # Run unsupervised clustering for white wine
    white_results = run_unsupervised_clustering(
        cleaned_csv=os.path.join(PROJECT_ROOT, "white_cleaned.csv"),
        prefix="white",
        out_dir=OUT_ROOT,
        seed=SEED
    )

    # Create quality-based heatmaps
    red_quality_centers = plot_quality_centers_heatmap(
        cleaned_csv=os.path.join(PROJECT_ROOT, "red_cleaned.csv"),
        prefix="red",
        out_dir=OUT_ROOT,
        use_zscore=True
    )

    white_quality_centers = plot_quality_centers_heatmap(
        cleaned_csv=os.path.join(PROJECT_ROOT, "white_cleaned.csv"),
        prefix="white",
        out_dir=OUT_ROOT,
        use_zscore=True
    )

    print(f"\n{'='*60}")
    print("All unsupervised and quality-based outputs saved!")
    print(f"Output directory: {OUT_ROOT}")
    print(f"{'='*60}")
