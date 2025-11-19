# DTI 5126 Wine Quality Analysis & Prediction

## Project Overview

This project performs comprehensive analysis and prediction of wine quality using both supervised and unsupervised machine learning approaches. The dataset combines red and white wine samples with physicochemical properties to predict quality ratings and discover natural groupings in the data.

## Project Structure

```
5170-Team-Project/
├── Data Exploration.R              # Exploratory data analysis
├── preprocessing_pipeline.R         # Data cleaning and feature engineering
├── supervised_model/
│   ├── Tree_model/
│   │   ├── FE+Supervised Tree models_final.R
│   │   ├── winequality-red.csv
│   │   └── winequality-white.csv
│   ├── Neural_Network/
│   │   ├── Deep Neural Network.py
│   │   └── wine_feature_engineered.csv
│   └── model_comparison.R          # Comprehensive model comparison & evaluation
└── unsupervised Cluster/
    ├── Unsupervised.R
    ├── red_cleaned.csv
    └── white_cleaned.csv
```

## Datasets

**Wine Quality Dataset**
- Red wine samples: physicochemical properties and quality ratings
- White wine samples: physicochemical properties and quality ratings
- Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol
- Target: quality (score 0-10)

## Workflow

### 1. Data Exploration (`Data Exploration.R`)

Initial exploratory data analysis to understand:
- Feature distributions
- Correlation patterns
- Missing values
- Class imbalance

### 2. Preprocessing (`preprocessing_pipeline.R`)

Data preparation pipeline:
- Duplicate removal
- Feature engineering:
  - `acidity_ratio = fixed.acidity / (volatile.acidity + citric.acid)`
  - `alcohol_va_ratio = alcohol / volatile.acidity`
  - `sugar_body_ratio = residual.sugar / (1/alcohol)`
  - `sulfur_efficiency = free.sulfur.dioxide / fixed.acidity`
- Binary label creation: quality ≥ 6 → "high", otherwise "low"
- Train/test split

### 3. Supervised Models

#### Tree-Based Models (`supervised_model/Tree_model/FE+Supervised Tree models_final.R`)

Implements and compares three tree-based classifiers:

**Decision Tree**
- Information gain splitting criterion
- Pruning based on cross-validation error
- Hyperparameters: `minbucket=50`, `maxdepth=7`, `cp=0.01`

**Random Forest**
- 500 trees
- `mtry = sqrt(p)` features per split
- Out-of-bag error estimation

**XGBoost**
- Gradient boosting with early stopping
- Hyperparameters: `max_depth=4`, `eta=0.1`, `subsample=0.8`, `colsample_bytree=0.8`
- 100 rounds with validation monitoring

**Evaluation:**
- 5-fold cross-validation
- Metrics: Accuracy, Precision, Recall, F1-Score, G-Mean
- Lift charts for model comparison
- Decision tree visualization

**How to run:**
```r
setwd("/path/to/5170-Team-Project/supervised_model/Tree_model")
source("FE+Supervised Tree models_final.R")
```

#### Deep Neural Network (`supervised_model/Neural_Network/Deep Neural Network.py`)

Feed-forward neural network for wine quality classification:

**Architecture:**
- Input layer: feature-engineered variables
- Hidden layers with ReLU activation and dropout
- Output layer: sigmoid activation for binary classification

**Training:**
- Optimizer: Adam
- Loss: Binary cross-entropy
- Regularization: Dropout layers

**How to run:**
```bash
cd supervised_model/Neural_Network
python "Deep Neural Network.py"
```

#### Model Comparison (`model_comparison.R`)

Comprehensive comparison framework for all supervised models:

**Features:**
- Unified evaluation pipeline across all models
- Side-by-side performance comparison
- Visual analytics:
  - Performance metrics bar charts
  - ROC curves comparison
  - Confusion matrix heatmaps
  - Feature importance plots
- Statistical significance testing
- Model ranking and recommendation

**Metrics Analyzed:**
- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC
- G-Mean
- Training time
- Prediction time

**How to run:**
```r
setwd("/path/to/5170-Team-Project/")
source("model_comparison.R")
```

### 4. Unsupervised Model (`unsupervised Cluster/Unsupervised.R`)

K-Means clustering to discover natural wine groupings:

**Methodology:**
- K-Means with k=2 clusters
- Feature scaling/standardization
- Cluster validation using silhouette scores
- Visualization of cluster separation

**Analysis:**
- Cluster profiling by quality distribution
- Feature importance per cluster
- Comparison of red vs. white wine clustering patterns

**How to run:**
```r
setwd("/path/to/5170-Team-Project/unsupervised Cluster")
source("Unsupervised.R")
```

## Dependencies

### R Packages
```r
install.packages(c(
  "readr", "dplyr", "caret", "rpart", "randomForest", 
  "xgboost", "gains", "rpart.plot", "ggplot2", "corrplot",
  "cluster", "reshape2", "pROC", "gridExtra"
))
```

### Python Packages
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
```

## Results

### Supervised Models Performance

Models are evaluated on:
- **Cross-Validation (5-fold):** Average performance across folds
- **Test Set:** Final holdout performance

Key metrics:
- Accuracy
- Precision & Recall (for "high" quality class)
- F1-Score
- G-Mean (geometric mean of sensitivity and specificity)
- AUC-ROC

The `model_comparison.R` script provides a comprehensive analysis with visualizations to identify the best performing model for wine quality prediction.

### Unsupervised Clustering

Discovers inherent groupings in wine samples based on physicochemical properties, providing insights into:
- Natural wine categories
- Feature patterns distinguishing clusters
- Relationship between clusters and quality ratings

## Team Members

SYS 5170 Applied Data Science - Team Project

## License

Educational project for SYS 5170 course.

