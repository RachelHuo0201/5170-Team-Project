 # Data Exploration
 
 # Import dataset
red <- read_delim("Downloads/wine+quality/winequality-red.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)
white <- read_delim("Downloads/wine+quality/winequality-white.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

 # ----- Wine Quality Distribution -----
library(ggplot2)

 # Compute counts per bin
red_counts <- as.data.frame(table(red$quality))
colnames(red_counts) <- c("quality", "count")
red_counts$quality <- as.numeric(as.character(red_counts$quality))
 
 # Histogram for red wine quality
ggplot(red, aes(x = quality)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  geom_text(data = red_counts,
            aes(x = quality, y = count, label = count),
            vjust = -0.5, color = "black") +
  labs(title = "Red Wine Quality Distribution",
       x = "Quality", y = "Count")

 # Compute counts per bin
white_counts <- as.data.frame(table(white$quality))
colnames(white_counts) <- c("quality", "count")
white_counts$quality <- as.numeric(as.character(white_counts$quality))

 # Histogram for white wine quality
ggplot(white, aes(x = quality)) +
  geom_histogram(binwidth = 1, fill = "gold", color = "black") +
  geom_text(data = white_counts,
            aes(x = quality, y = count, label = count),
            vjust = -0.5, color = "black") +
  labs(title = "White Wine Quality Distribution",
       x = "Quality", y = "Count")

# ----- Wine Features Correlation Heatmap -----
library(corrplot)

# Red Wine Correlation Heatmap
# Compute correlation matrix
red_cor <- cor(red)

# Plot heatmap
corrplot(red_cor,
         method = "color",
         type = "lower",
         addCoef.col = "black",   # add numeric values
         tl.col = "black",        # label color
         tl.srt = 45,             # rotate labels
         number.cex = 0.7,        # size of numbers
         col = colorRampPalette(c("#313695", "#4575B4", "#74ADD1", "#ABD9E9",
                                  "#E0F3F8", "#FFFFBF", "#FEE090", "#FDAE61",
                                  "#F46D43", "#D73027", "#A50026"))(200),
         title = "Red Wine Features Correlation Heatmap",
         mar = c(0,0,2,0)
)

# White Wine Correlation Heatmap
# Compute correlation matrix
white_cor <- cor(white)

# Plot heatmap
corrplot(white_cor,
         method = "color",
         type = "lower",
         addCoef.col = "black", 
         tl.col = "black",
         tl.srt = 45,
         number.cex = 0.7,
         col = colorRampPalette(c("#313695", "#4575B4", "#74ADD1", "#ABD9E9",
                                  "#E0F3F8", "#FFFFBF", "#FEE090", "#FDAE61",
                                  "#F46D43", "#D73027", "#A50026"))(200),
         title = "White Wine Features Correlation Heatmap",
         mar = c(0,0,2,0)
)

library(dplyr)
library(ggplot2)
library(tidyr)
library(scales)

# -------Top 5 correlated Features(Scaled Z-scores)---------
# 1. Compute correlations
# -----------------------------
cor_values <- cor(red)[, "quality"]
cor_values <- cor_values[names(cor_values) != "quality"]

# Top 5 by absolute correlation
top5 <- sort(abs(cor_values), decreasing = TRUE)[1:5]
top5_names <- names(top5)

# Extract signs
cor_sign <- cor_values[top5_names]

# -----------------------------
# 2. Prepare dataframe
# -----------------------------
df_top5 <- red %>%
  select(all_of(top5_names)) %>%
  mutate(across(everything(), scale)) %>%     # scale to Z-scores
  mutate(id = row_number()) %>%
  pivot_longer(cols = -id, names_to = "feature", values_to = "zscore")

# Add sign info for coloring
df_top5 <- df_top5 %>%
  mutate(sign = ifelse(cor_sign[feature] > 0, "Positive", "Negative"))

# -----------------------------
# 3. Plot Red Wine
# -----------------------------
ggplot(df_top5, aes(x = feature, y = zscore, fill = sign)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  scale_fill_manual(values = c("Positive" = "#2E8B57", "Negative" = "#D46A6A")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Standardized Distributions of Key Predictors for Red Wine Quality",
    subtitle = "Features scaled to Z-scores (mean = 0, SD = 1) for fair comparison",
    x = "",
    y = "Standardized Value (Z-score)",
    fill = "Relationship with Quality"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12)
  ) +
  # correlation labels under each box
  geom_text(
    data = data.frame(
      feature = top5_names,
      r_label = paste0("r = ", round(cor_sign, 3)),
      y = -3.2,
      sign = ifelse(cor_sign > 0, "Positive", "Negative")
    ),
    aes(x = feature, y = y, label = r_label, color = sign),
    size = 4
  ) +
  scale_color_manual(values = c("Positive" = "#2E8B57", "Negative" = "#D46A6A"), guide = "none")

library(dplyr)
library(ggplot2)
library(tidyr)

# -----------------------------
# 1. Compute correlations
# -----------------------------
cor_values_white <- cor(white)[, "quality"]
cor_values_white <- cor_values_white[names(cor_values_white) != "quality"]

# Top 5 features by absolute correlation
top5_white <- sort(abs(cor_values_white), decreasing = TRUE)[1:5]
top5_white_names <- names(top5_white)

# Extract actual correlation signs
cor_sign_white <- cor_values_white[top5_white_names]

# -----------------------------
# 2. Prepare dataframe
# -----------------------------
df_top5_white <- white %>%
  select(all_of(top5_white_names)) %>%
  mutate(across(everything(), scale)) %>%   # Z-scores
  mutate(id = row_number()) %>%
  pivot_longer(cols = -id, names_to = "feature", values_to = "zscore")

# Add sign indicator for color
df_top5_white <- df_top5_white %>%
  mutate(sign = ifelse(cor_sign_white[feature] > 0, "Positive", "Negative"))

# -----------------------------
# 3. Plot White Wine
# -----------------------------
ggplot(df_top5_white, aes(x = feature, y = zscore, fill = sign)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  scale_fill_manual(values = c("Positive" = "#2E8B57", "Negative" = "#D46A6A")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Standardized Distributions of Key Predictors for White Wine Quality",
    subtitle = "Features scaled to Z-scores (mean = 0, SD = 1) for fair comparison",
    x = "",
    y = "Standardized Value (Z-score)",
    fill = "Relationship with Quality"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12)
  ) +
  geom_text(
    data = data.frame(
      feature = top5_white_names,
      r_label = paste0("r = ", round(cor_sign_white, 3)),
      y = -3.2,
      sign = ifelse(cor_sign_white > 0, "Positive", "Negative")
    ),
    aes(x = feature, y = y, label = r_label, color = sign),
    size = 4
  ) +
  scale_color_manual(values = c("Positive" = "#2E8B57", "Negative" = "#D46A6A"), guide = "none")

library(GGally)

# ----- Key Features Correlation Matrix -----
# Select features
red_features <- red %>%
  select(alcohol, volatile.acidity, sulphates, citric.acid, total.sulfur.dioxide, quality)

white_features <- white %>%
  select(alcohol, density, chlorides, volatile.acidity, total.sulfur.dioxide, quality)

# Red wine plot
ggpairs(
  red_features,
  title = "Red Wine: Key Features Relationships",
  upper = list(continuous = wrap("cor", size = 4)),
  diag = list(continuous = wrap("densityDiag")),
  lower = list(continuous = wrap("points", alpha = 0.3, size = 0.5))
)

# White wine plot
ggpairs(
  white_features,
  title = "White Wine: Key Features Relationships",
  upper = list(continuous = wrap("cor", size = 4)),
  diag = list(continuous = wrap("densityDiag")),
  lower = list(continuous = wrap("points", alpha = 0.3, size = 0.5))
)

