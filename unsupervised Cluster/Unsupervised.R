# ================================================================
# Unsupervised K-Means Clustering & Visualization
# ================================================================

# --- Configuration (only for unsupervised) ---
project_root <- "."   

cleaned_dir <- project_root                              
out_root    <- file.path(project_root, "out")  
seed <- 42

library(ggplot2)
library(cluster)
library(reshape2)

# === Global feature order (for consistent heatmap comparison) ===
base_df <- read.csv(file.path(cleaned_dir, "red_cleaned.csv"), stringsAsFactors = FALSE)
global_features <- names(base_df)[sapply(base_df, is.numeric)]
global_features <- setdiff(global_features, "quality")


# ================================================================
# 1. Unsupervised K-Means clustering and visualization
# ================================================================
run_unsupervised_simple <- function(cleaned_csv, prefix, out_dir = ".", seed = 42) {
  set.seed(seed)
  
  # --- Load cleaned dataset ---
  data_clean <- read.csv(cleaned_csv, stringsAsFactors = FALSE)
  
  # --- Select numeric features (exclude quality) ---
  num_cols <- names(data_clean)[sapply(data_clean, is.numeric)]
  num_cols <- setdiff(num_cols, c("quality"))
  X <- data_clean[, num_cols, drop = FALSE]
  Xs <- scale(X)   # if after normalization as.matrix(X)
  
  # --- Determine optimal k (2–8) using mean silhouette width ---
  ks <- 2:8
  sils <- numeric(length(ks))
  for (i in seq_along(ks)) {
    k <- ks[i]
    km <- kmeans(Xs, centers = k, nstart = 25, iter.max = 100)
    sil <- silhouette(km$cluster, dist(Xs))
    sils[i] <- mean(sil[, "sil_width"])
  }
  k_best <- ks[which.max(sils)]
  message(sprintf("[%s] Best k = %d (mean silhouette = %.3f)", prefix, k_best, max(sils)))
  
  # --- Fit final model using the best k ---
  km_best <- kmeans(Xs, centers = k_best, nstart = 25, iter.max = 200)
  clusters <- factor(km_best$cluster)
  
  # --- Output directory for unsupervised results ---
  out_dir2 <- out_dir
  if (!dir.exists(out_dir2)) dir.create(out_dir2, recursive = TRUE)
  
  # --- Silhouette curve ---
  df_k <- data.frame(k = ks, Silhouette = sils)
  g_sil <- ggplot(df_k, aes(k, Silhouette)) +
    geom_line() + geom_point() +
    theme_minimal() +
    labs(title = paste0(prefix, " - Mean Silhouette (k selection)"),
         x = "Number of clusters (k)", y = "Mean silhouette width")
  ggsave(file.path(out_dir2, paste0(prefix, "_silhouette.png")), g_sil, width = 5.5, height = 4)
  
  # --- PCA plot (2D visualization of final clusters) ---
  pca <- prcomp(Xs, center = TRUE, scale. = TRUE)
  pca_df <- data.frame(
    PC1 = pca$x[,1],
    PC2 = pca$x[,2],
    cluster = clusters
  )
  
  p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
    geom_point(alpha = 0.6, size = 2) +
    theme_minimal() +
    labs(
      title = paste0(prefix, " - PCA of Final Clusters (k = ", k_best, ")"),
      x = "PC1",
      y = "PC2",
      color = "Cluster"
    )
  
  ggsave(file.path(out_dir2, paste0(prefix, "_cluster_pca.png")),
         p_pca, width = 6.5, height = 5)
  message("Saved PCA cluster plot.")
  
  
  # --- Cluster centers heatmap ---
  centers_df <- as.data.frame(km_best$centers)
  centers_df$cluster <- factor(seq_len(nrow(centers_df)), levels = rev(seq_len(nrow(centers_df))))
  feat_names <- setdiff(names(centers_df), "cluster")
  centers_long <- do.call(rbind, lapply(seq_len(nrow(centers_df)), function(i) {
    data.frame(
      cluster = centers_df$cluster[i],
      feature = feat_names,
      mean_z = as.numeric(centers_df[i, feat_names]),
      stringsAsFactors = FALSE
    )
  }))
  
  centers_long$feature <- factor(centers_long$feature, levels = global_features)
  
  g_heat <- ggplot(centers_long, aes(x = feature, y = cluster, fill = mean_z)) +
    geom_tile() +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5)
    ) +
    labs(title = paste0(prefix, " - Cluster Centers (z-scored Feature Means)"),
         x = NULL, y = "Cluster", fill = "Mean (z)")
  ggsave(file.path(out_dir2, paste0(prefix, "_centers_heatmap.png")),
         g_heat, width = 7.5, height = 4.8)
  
  # --- Cross-tab: cluster vs quality ---
  cross_tab_num <- as.data.frame.matrix(table(cluster = clusters, quality = data_clean$quality))
  write.csv(cross_tab_num, file.path(out_dir2, paste0(prefix, "_cluster_vs_quality_numeric.csv")), row.names = TRUE)
  
  # --- Column-wise percentages (each column sums to 100%) ---
  cross_tab_pct <- sweep(cross_tab_num, 2, colSums(cross_tab_num), FUN = "/") * 100
  cross_tab_pct <- round(cross_tab_pct, 1)
  write.csv(cross_tab_pct, file.path(out_dir2, paste0(prefix, "_cluster_vs_quality_percent.csv")), row.names = TRUE)
  
  # --- Heatmap of cluster vs quality (% per column) ---
  df_prop <- cross_tab_pct
  df_prop$cluster <- rownames(df_prop)
  df_long <- melt(df_prop, id.vars = "cluster", variable.name = "quality", value.name = "percent")
  df_long$quality <- gsub("^X", "", trimws(df_long$quality))
  df_long$quality <- suppressWarnings(as.numeric(df_long$quality))
  df_long <- df_long[!is.na(df_long$quality), ]
  df_long$quality <- factor(df_long$quality, levels = sort(unique(df_long$quality)))
  df_long$cluster <- factor(df_long$cluster, levels = rev(unique(df_long$cluster)))
  
  p <- ggplot(df_long, aes(x = quality, y = cluster, fill = percent)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.1f%%", percent)), size = 3) +
    scale_fill_gradient(limits = c(0, 100), low = "lightblue", high = "red") +
    labs(title = paste0(prefix, " – Cluster vs Quality (%)"),
         x = "Quality (1–9)", y = "Cluster", fill = "Proportion (%)") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(file.path(out_dir2, paste0(prefix, "_cluster_quality_heatmap_percent.png")),
         p, width = 6, height = 4.5)
}


# ================================================================
# 2. Quality-based Feature Means Heatmap (for supervised comparison)
# ================================================================
plot_quality_centers_heatmap <- function(
    cleaned_csv, prefix, out_dir = ".", use_zscore = TRUE, label_cells = TRUE) {
  
  df <- read.csv(cleaned_csv, stringsAsFactors = FALSE)
  num_cols <- names(df)[sapply(df, is.numeric)]
  num_cols <- setdiff(num_cols, "quality")
  if (length(num_cols) == 0) stop("No numeric feature columns found.")
  
  X <- df[, num_cols, drop = FALSE]
  Xs <- if (isTRUE(use_zscore)) scale(X) else as.matrix(X)
  
  grp <- factor(df$quality, levels = sort(unique(df$quality)))
  agg <- aggregate(Xs, by = list(Quality = grp), FUN = mean, na.rm = TRUE)
  
  long <- melt(agg, id.vars = "Quality", variable.name = "feature", value.name = "mean_val")
  long$feature <- factor(long$feature, levels = global_features)
  
  p <- ggplot(long, aes(x = feature, y = Quality, fill = mean_val)) +
    geom_tile() +
    { if (label_cells) geom_text(aes(label = sprintf("%.2f", mean_val)), size = 3, color = "black") } +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste0(prefix, " - Quality Group Centers (z-scored Feature Means)"),
         x = "Feature", y = "Quality", fill = "Mean (z)")
  
  out_dir2 <- out_dir
  if (!dir.exists(out_dir2)) dir.create(out_dir2, recursive = TRUE)
  fname <- file.path(out_dir2, paste0(prefix, "_quality_centers_quality_z.png"))
  ggsave(fname, p, width = 7.5, height = 4.8)
  message("Saved quality group centers heatmap: ", normalizePath(fname, winslash = "/"))
}


# ================================================================
# 3. Run all experiments 
# ================================================================
run_unsupervised_simple(
  cleaned_csv = file.path(cleaned_dir, "red_cleaned.csv"),
  prefix = "red",
  out_dir = out_root,
  seed = seed
)

run_unsupervised_simple(
  cleaned_csv = file.path(cleaned_dir, "white_cleaned.csv"),
  prefix = "white",
  out_dir = out_root,
  seed = seed
)

plot_quality_centers_heatmap(
  cleaned_csv = file.path(cleaned_dir, "red_cleaned.csv"),
  prefix = "red",
  out_dir = out_root,
  use_zscore = TRUE, label_cells = TRUE
)

plot_quality_centers_heatmap(
  cleaned_csv = file.path(cleaned_dir, "white_cleaned.csv"),
  prefix = "white",
  out_dir = out_root,
  use_zscore = TRUE, label_cells = TRUE
)

message("All unsupervised and quality-based outputs saved under: ",
        file.path(out_root, "unsupervised"))
