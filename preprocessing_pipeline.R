# === Configuration ===
data_dir <- "."                # Input data folder
out_dir <- "."             # Output directory
dataset_mode <- 'both'                                          # 'red', 'white', or 'both'
seed <- 42                                                      # Random seed for reproducibility
do_scale <- TRUE                                                # Whether to standardize numeric predictors

library(ggplot2)



# ================================================================
# 1. Basic data preprocessing
# ================================================================
process_simple <- function(path, prefix, out_dir = '.', scale = TRUE, seed = 42) {
  message('Processing dataset: ', prefix)
  df <- read.csv(path, sep = ';', header = TRUE, stringsAsFactors = FALSE)
  colnames(df) <- c(
    'fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides',
    'free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality'
  )
  
  # --- Missing values and duplicates ---
  print(colSums(is.na(df)))
  message('Duplicates before: ', sum(duplicated(df)))
  df <- df[!duplicated(df), ]
  message('Duplicates after: ', sum(duplicated(df)))
  
  # --- Count outliers using the IQR method ---
  num_cols <- names(df)[sapply(df, is.numeric)]
  outlier_counts <- data.frame(variable = character(), outliers = numeric(), stringsAsFactors = FALSE)
  for (col in num_cols) {
    x <- df[[col]]
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQRv <- Q3 - Q1
    lower <- Q1 - 1.5 * IQRv
    upper <- Q3 + 1.5 * IQRv
    n_out <- sum(x < lower | x > upper, na.rm = TRUE)
    outlier_counts <- rbind(outlier_counts, data.frame(variable = col, outliers = n_out))
  }
  message('Outlier counts by feature:')
  print(outlier_counts)
  
  # --- Create quality label (low / medium / high) ---
  df$quality <- as.numeric(df$quality)
  df$quality_label <- factor(
    ifelse(df$quality >= 7, 'high',
           ifelse(df$quality == 6, 'medium', 'low')),
    levels = c('low','medium','high')
  )
  
  # --- Optionally scale numeric predictors (except quality) ---
  if (isTRUE(scale)) {
    num_features <- names(df)[sapply(df, is.numeric) & names(df) != 'quality']
    df[num_features] <- scale(df[num_features])
  }
  
  # --- Train/test split (80/20) ---
  set.seed(seed)
  n <- nrow(df)
  test_idx <- sample(seq_len(n), size = floor(0.2 * n))
  train <- df[-test_idx, ]
  test <- df[test_idx, ]
  
  # --- Save cleaned outputs ---
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  write.csv(df, file = file.path(out_dir, paste0(prefix, '_cleaned.csv')), row.names = FALSE)
  write.csv(train, file = file.path(out_dir, paste0(prefix, '_train.csv')), row.names = FALSE)
  write.csv(test, file = file.path(out_dir, paste0(prefix, '_test.csv')), row.names = FALSE)
  write.csv(outlier_counts, file = file.path(out_dir, paste0(prefix, '_outlier_counts.csv')), row.names = FALSE)
  
  # --- Plot outlier counts per feature ---
  plots_dir <- file.path(out_dir, 'plots')
  if (!dir.exists(plots_dir)) dir.create(plots_dir)
  ggsave(
    filename = file.path(plots_dir, paste0(prefix, '_outlier_counts.png')),
    plot = ggplot(outlier_counts, aes(x = reorder(variable, -outliers), y = outliers)) +
      geom_bar(stat = 'identity', fill = 'steelblue') + theme_minimal(),
    width = 8, height = 3
  )
  
  message('Saved cleaned data to: ', normalizePath(out_dir))
}

if (dataset_mode %in% c('red','both'))
  process_simple(file.path(data_dir, 'winequality-red.csv'), 'red', out_dir = out_dir, scale = do_scale, seed = seed)
if (dataset_mode %in% c('white','both'))
  process_simple(file.path(data_dir, 'winequality-white.csv'), 'white', out_dir = out_dir, scale = do_scale, seed = seed)

message('Data preprocessing completed.')