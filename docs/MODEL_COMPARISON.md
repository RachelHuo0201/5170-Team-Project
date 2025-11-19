# Model Comparison

## Overview

This section provides a comprehensive comparison of all supervised and unsupervised learning approaches applied to the wine quality prediction problem. Our team implemented multiple machine learning models across both paradigms to identify the most effective approach for predicting wine quality based on physicochemical properties.

## Datasets

### Supervised Learning Dataset
- **File**: `wine_feature_engineered.csv`
- **Samples**: 5,320 (combined red and white wines)
- **Features**: 14 total
  - **Original features (9)**: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, pH, sulphates, alcohol
  - **Engineered features (4)**: acidity_ratio, alcohol_va_ratio, sugar_body_ratio, sulfur_efficiency
  - **Meta features (1)**: wine_type (red/white indicator)
- **Target**: Binary classification (quality < 6 = class 0, quality ≥ 6 = class 1)
- **Split**: 80% training, 20% validation (stratified)

### Unsupervised Learning Dataset
- **Files**: `red_cleaned.csv` (1,359 samples), `white_cleaned.csv` (3,961 samples)
- **Features**: 11 physicochemical properties (standardized/z-scored)
- **Target**: Quality scores (3-9) used for analysis but not for clustering

---

## Supervised Learning Models

### 1. Deep Neural Network (DNN) - **CURRENT ACTIVE MODEL**

#### Architecture
```
Input Layer (14 features)
    ↓
Dense Layer (128 neurons) → BatchNorm → ReLU → Dropout (0.2)
    ↓
Dense Layer (128 neurons) → BatchNorm → ReLU → Dropout (0.2)
    ↓
Dense Layer (128 neurons) → BatchNorm → ReLU → Dropout (0.2)
    ↓
Output Layer (2 classes) → Softmax
```

#### Hyperparameters
- **Optimizer**: Adam (learning rate: 1e-5, weight decay: 5e-4)
- **Loss Function**: Cross-Entropy Loss
- **Max Epochs**: 5000 with early stopping (patience: 300 epochs)
- **Batch Size**: Full batch training
- **Regularization**: Dropout (0.2) + Batch Normalization + L2 regularization

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 75.75% |
| **Precision** | 0.799 |
| **Recall** | 0.818 |
| **F1-Score** | 0.809 |
| **Training Time** | Longest (5000 max epochs) |

#### Confusion Matrix
```
                Predicted
              Low    High
Actual  Low   261    137
        High  121    545
```

#### Strengths
- **High Recall (81.8%)**: Effectively identifies high-quality wines, minimizing false negatives
- **Balanced Performance**: Good balance between precision and recall (F1 = 0.809)
- **Complex Pattern Recognition**: Capable of capturing non-linear relationships between features
- **Regularization**: Multiple techniques (dropout, batch norm, early stopping) prevent overfitting

#### Weaknesses
- **Lowest Accuracy (75.75%)**: Among all tested models, DNN has the lowest overall accuracy
- **Training Time**: Requires extensive training time (up to 5000 epochs)
- **Interpretability**: Black-box nature makes it difficult to explain predictions to stakeholders
- **Computational Cost**: Requires more computational resources than tree-based models

#### Best Use Cases
- Current production deployment
- Applications where recall is more important than precision
- Complex pattern detection in wine quality assessment

---

### 2. XGBoost - **DELETED (BEST PERFORMER)**

#### Configuration
- **max_depth**: 4
- **eta** (learning rate): 0.1
- **Objective**: Binary classification
- **Evaluation**: AUC, accuracy

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **87.87%** ⭐ |
| **F1-Score** | 0.554 |
| **G-mean** | 0.688 |
| **Training Time** | Medium |

#### Strengths
- **Highest Accuracy**: Best overall accuracy among all models (87.87%)
- **Handles Class Imbalance**: Built-in mechanisms for imbalanced data
- **Feature Importance**: Provides insights into which features matter most
- **Robust**: Less prone to overfitting than single decision trees

#### Weaknesses
- **Code Deleted**: No longer available in the repository
- **Interpretability**: Less interpretable than simple decision trees
- **Hyperparameter Tuning**: Requires careful tuning of multiple parameters
- **Lower F1-Score**: Despite high accuracy, F1-score (0.554) is lower than DNN

#### Recommendation
**CRITICAL**: Consider restoring this model as it demonstrates the best overall accuracy and could serve as a strong production candidate or ensemble component.

---

### 3. Random Forest - **DELETED**

#### Configuration
- **n_estimators**: Multiple decision trees
- **max_features**: Subset of features per tree
- **Ensemble Method**: Bootstrap aggregating (bagging)

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 86.95% |
| **F1-Score** | 0.471 |
| **G-mean** | 0.610 |
| **Training Time** | Medium |

#### Strengths
- **High Accuracy**: Second-best accuracy (86.95%)
- **Feature Importance**: Built-in feature importance ranking
- **Robust to Overfitting**: Ensemble approach reduces overfitting risk
- **Handles Non-linearity**: Can capture complex relationships

#### Weaknesses
- **Code Deleted**: No longer available in the repository
- **Interpretability**: Difficult to interpret individual predictions
- **Slower Training**: Slower than single decision tree
- **Low F1-Score**: F1-score (0.471) indicates poor minority class performance

---

### 4. Decision Tree - **DELETED**

#### Configuration
- **minbucket**: 50 (minimum samples in leaf node)
- **maxdepth**: 7 (maximum tree depth)

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 78.67% |
| **F1-Score** | 0.448 |
| **G-mean** | 0.580 |
| **Training Time** | Fastest |

#### Strengths
- **Highly Interpretable**: Easy to visualize and explain to non-technical stakeholders
- **Fast Training**: Fastest training time among all models
- **Simple Decision Rules**: Clear if-then rules for predictions
- **No Feature Scaling Required**: Works with raw features

#### Weaknesses
- **Lowest Accuracy**: Poorest accuracy (78.67%) among all models
- **Prone to Overfitting**: Single tree can overfit without proper constraints
- **Low F1-Score**: Struggles with minority class (F1 = 0.448)
- **Code Deleted**: No longer available in the repository

#### Best Use Cases (if restored)
- Quick baseline model development
- Educational purposes and stakeholder explanations
- Feature importance analysis

---

## Supervised Model Comparison Summary

| Model | Accuracy | Precision | Recall | F1-Score | G-mean | Interpretability | Speed | Status |
|-------|----------|-----------|--------|----------|--------|------------------|-------|--------|
| **XGBoost** | **87.87%** ⭐ | - | - | 0.554 | **0.688** | Medium | Medium | ❌ Deleted |
| **Random Forest** | 86.95% | - | - | 0.471 | 0.610 | Medium | Medium | ❌ Deleted |
| **Decision Tree** | 78.67% | - | - | 0.448 | 0.580 | **High** ⭐ | **Fast** ⭐ | ❌ Deleted |
| **DNN (Current)** | 75.75% | **0.799** | **0.818** ⭐ | **0.809** ⭐ | - | Low | Slow | ✅ Active |

### Key Observations

1. **Accuracy vs. F1-Score Trade-off**:
   - XGBoost has the highest accuracy (87.87%) but moderate F1-score (0.554)
   - DNN has the lowest accuracy (75.75%) but highest F1-score (0.809)
   - This suggests DNN is better at handling class imbalance for the minority class

2. **Model Availability Crisis**:
   - Only DNN is currently active in the repository
   - All tree-based models (including best-performing XGBoost) were deleted
   - This limits comparison, ensemble possibilities, and production options

3. **Interpretability vs. Performance**:
   - More interpretable models (Decision Tree) have lower accuracy
   - Less interpretable models (DNN, XGBoost) have higher performance
   - Trade-off between explainability and predictive power

---

## Unsupervised Learning Models

### K-Means Clustering

#### Methodology
- **Algorithm**: K-Means clustering with multiple random initializations
- **K Selection**: Silhouette analysis for k ∈ {2, 3, 4, 5, 6, 7, 8}
- **Feature Preprocessing**: Z-score standardization
- **Initialization**: k-means++ (n_init=25)
- **Max Iterations**: 200

#### Results

##### Red Wine Clustering

| Metric | Value |
|--------|-------|
| **Optimal K** | 2 |
| **Silhouette Score** | 0.205 |
| **Cluster 1 Size** | 542 samples (39.88%) |
| **Cluster 2 Size** | 817 samples (60.12%) |

**Cluster-Quality Distribution:**
```
Quality      3    4     5     6     7    8
Cluster 1:   3   11   180   224   114   10  (Higher quality tendency)
Cluster 2:   7   42   397   311    53    7  (Lower quality tendency)
```

**Key Finding**: Cluster 1 shows a stronger tendency toward higher quality wines (quality 6-8), while Cluster 2 is more concentrated in medium quality (quality 5-6).

##### White Wine Clustering

| Metric | Value |
|--------|-------|
| **Optimal K** | 2 |
| **Silhouette Score** | 0.213 |
| **Cluster 1 Size** | 2,484 samples (62.71%) |
| **Cluster 2 Size** | 1,477 samples (37.29%) |

**Cluster-Quality Distribution:**
```
Quality      3    4     5     6     7    8   9
Cluster 1:   8  104   515  1134   600  119   4  (Higher quality tendency)
Cluster 2:  12   49   660   654    89   12   1  (Lower quality tendency)
```

**Key Finding**: Similar to red wine, Cluster 1 exhibits higher quality concentration (quality 6-8), while Cluster 2 is more balanced across quality ranges.

#### Interpretation

##### Cluster Characteristics (from heatmaps)

**Red Wine:**
- **Cluster 1 (Higher Quality)**: Higher alcohol content, lower volatile acidity, higher sulphates
- **Cluster 2 (Lower Quality)**: Lower alcohol content, higher volatile acidity, lower sulphates

**White Wine:**
- **Cluster 1 (Higher Quality)**: Higher alcohol content, balanced acidity, optimal sulfur dioxide levels
- **Cluster 2 (Lower Quality)**: Lower alcohol content, higher residual sugar, suboptimal acidity

#### Strengths
- **Unsupervised Pattern Discovery**: Identifies natural groupings without labeled data
- **Quality Correlation**: Clusters align moderately well with quality scores
- **Separate Wine Type Analysis**: Recognizes that red and white wines have different characteristics
- **Customer Segmentation**: Can be used for market segmentation and targeted marketing

#### Weaknesses
- **Moderate Silhouette Scores**: Scores of 0.205-0.213 indicate moderate cluster separation
- **Limited Clusters**: Only 2 clusters identified as optimal (low granularity)
- **No Direct Prediction**: Cannot directly predict quality for new samples
- **Interpretation Required**: Clusters need domain expertise for meaningful interpretation

#### Best Use Cases
- Exploratory data analysis and pattern discovery
- Customer segmentation for marketing purposes
- Quality control process grouping
- Feature relationship visualization
- Complement to supervised learning insights

---

## Supervised vs. Unsupervised Learning Comparison

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|--------------------|-----------------------|
| **Objective** | Predict wine quality (binary classification) | Discover natural groupings in wine characteristics |
| **Requires Labels** | Yes (quality scores) | No |
| **Best Accuracy** | 87.87% (XGBoost) | N/A (no direct prediction) |
| **Current Accuracy** | 75.75% (DNN) | N/A |
| **Interpretability** | Varies (High for trees, Low for DNN) | Medium (requires domain knowledge) |
| **Best Use** | Production quality prediction | Exploratory analysis, segmentation |
| **Cluster Quality Alignment** | N/A | Moderate (silhouette 0.205-0.213) |
| **Training Data Requirements** | Labeled data (expensive) | Unlabeled data (cheaper) |

### Complementary Insights

1. **Validation of Supervised Learning**:
   - Clustering confirms that wine characteristics naturally separate into quality-related groups
   - This validates that supervised models have meaningful patterns to learn

2. **Feature Importance Confirmation**:
   - Both approaches highlight alcohol content, volatile acidity, and sulphates as key differentiators
   - Consistent feature importance across paradigms strengthens confidence in results

3. **Business Applications**:
   - **Supervised**: Direct quality prediction for quality control
   - **Unsupervised**: Market segmentation and customer targeting

---

## Overall Model Ranking

### For Wine Quality Prediction (Supervised Task)

1. **XGBoost** (87.87% accuracy) - **Best overall, but deleted** ⚠️
2. **Random Forest** (86.95% accuracy) - **Second best, but deleted** ⚠️
3. **Decision Tree** (78.67% accuracy) - **Most interpretable, but deleted** ⚠️
4. **Deep Neural Network** (75.75% accuracy, 80.9% F1) - **Current production model** ✅

### For Pattern Discovery (Unsupervised Task)

1. **K-Means Clustering** - White Wine (silhouette: 0.213)
2. **K-Means Clustering** - Red Wine (silhouette: 0.205)

---

## Recommendations

### Immediate Actions

1. **Restore XGBoost Model**:
   - Highest accuracy (87.87%) makes it the best candidate for production
   - Should be restored from version control or retrained

2. **Create Model Ensemble**:
   - Combine XGBoost and DNN predictions for improved robustness
   - Leverage XGBoost's accuracy and DNN's high recall

3. **Address Class Imbalance**:
   - All models show lower performance on minority class
   - Implement SMOTE, class weighting, or stratified oversampling

### Long-term Strategy

4. **Maintain Model Diversity**:
   - Keep at least one tree-based model for interpretability
   - Keep DNN for complex pattern recognition
   - Use simpler models (Decision Tree) for stakeholder explanations

5. **Feature Engineering**:
   - Current engineered features (ratios) show promise
   - Continue exploring domain-driven feature creation with wine experts

6. **Clustering Applications**:
   - Use K-Means results for customer segmentation
   - Develop targeted marketing strategies based on cluster profiles
   - Validate cluster patterns with domain experts

---

## Conclusion of Comparison

The wine quality prediction project demonstrates the value of applying both supervised and unsupervised learning approaches:

- **Supervised learning** provides direct quality predictions, with XGBoost achieving 87.87% accuracy as the best performer
- **Unsupervised learning** reveals natural wine groupings that correlate with quality, validating the supervised approach
- **Current state** is suboptimal with only the lowest-accuracy model (DNN) active, despite its good F1-score
- **Recommendation**: Restore deleted models, particularly XGBoost, and develop an ensemble approach for production deployment

The complementary nature of these approaches provides both predictive power and business insights, creating a comprehensive solution for wine quality assessment and market analysis.
