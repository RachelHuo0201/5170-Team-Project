# Wine Quality Prediction using Machine Learning

## SYS 5170 Team Project

A comprehensive machine learning project applying both **supervised** and **unsupervised learning** approaches to predict wine quality based on physicochemical properties.

---

## 📊 Project Overview

This project demonstrates the application of multiple machine learning techniques to predict wine quality from objective physicochemical measurements. We achieved **87.87% accuracy** using XGBoost and discovered natural quality-related groupings through K-Means clustering.

### Key Achievements

- ✅ **87.87% accuracy** with XGBoost (best performer)
- ✅ **80.9% F1-score** with Deep Neural Network (best minority class handling)
- ✅ Identified **2 natural clusters** in wine characteristics correlating with quality
- ✅ Created **engineered features** that improved model performance
- ✅ Comprehensive **model comparison** across 4 supervised and 2 unsupervised approaches

---

## 📁 Project Structure

```
5126-Team-Project/
│
├── supervised_model/                  # Supervised learning implementations
│   └── Neural_Network/
│       ├── Deep Neural Network.py     # Current active model (DNN)
│       ├── NN_confusion_matrix.png    # Confusion matrix visualization
│       ├── NN_accuracy.png            # Training/validation accuracy curves
│       ├── NN_loss.png                # Loss convergence plot
│       ├── NN_F1.png                  # F1-score progression
│       └── wine_feature_engineered.csv # Engineered dataset (5,320 samples)
│
├── unsupervised Cluster/              # Unsupervised learning implementations
│   ├── Unsupervised.R                 # Original R clustering script
│   ├── unsupervised_analysis.py       # Python implementation (NEW)
│   ├── red_cleaned.csv                # Red wine data (1,359 samples)
│   ├── white_cleaned.csv              # White wine data (3,961 samples)
│   └── out/                           # Clustering results (NEW)
│       ├── red_silhouette.png
│       ├── red_cluster_pca.png
│       ├── red_centers_heatmap.png
│       ├── red_cluster_quality_heatmap_percent.png
│       ├── red_quality_centers_quality_z.png
│       └── white_* (same visualizations for white wine)
│
├── model_comparison/                   # Comprehensive model analysis (NEW)
│   ├── comparison_analysis.py          # Analysis script
│   └── results/
│       ├── supervised_models_comparison.csv
│       ├── unsupervised_models_comparison.csv
│       ├── model_performance_summary.csv
│       ├── key_insights.csv
│       ├── model_accuracy_comparison.png
│       ├── model_f1_comparison.png
│       ├── dnn_metrics_radar.png
│       ├── clustering_quality_distribution.png
│       ├── key_insights_summary.png
│       └── model_tradeoffs.png
│
├── docs/                               # Comprehensive documentation (NEW)
│   ├── MODEL_COMPARISON.md             # Detailed model comparison
│   ├── CONCLUSION.md                   # Project conclusion
│   └── DISCUSSION.md                   # In-depth discussion and analysis
│
└── README.md                           # This file
```

---

## 🎯 Problem Statement

**Objective**: Predict wine quality from physicochemical properties to:
- Reduce costs and time in quality assessment
- Enable data-driven winemaking process optimization
- Support quality control and consistency assurance
- Enable customer segmentation and targeted marketing

**Dataset**: UCI Machine Learning Repository - Wine Quality Dataset
- **Red Wine**: 1,359 samples
- **White Wine**: 3,961 samples
- **Total**: 5,320 samples (combined for supervised learning)
- **Features**: 11 physicochemical properties + 4 engineered features
- **Target**: Quality scores (3-9) → Binary classification (< 6 vs. ≥ 6)

---

## 📈 Results Summary

### Supervised Learning Models

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| **XGBoost** | **87.87%** ⭐ | - | - | 0.554 | ❌ Deleted |
| **Random Forest** | 86.95% | - | - | 0.471 | ❌ Deleted |
| **Decision Tree** | 78.67% | - | - | 0.448 | ❌ Deleted |
| **Deep Neural Network** | 75.75% | 0.799 | **0.818** ⭐ | **0.809** ⭐ | ✅ Active |

**Key Finding**: XGBoost achieved the highest accuracy but was deleted. DNN has the best F1-score (minority class handling) but lowest accuracy.

### Unsupervised Learning Results

| Wine Type | Optimal K | Silhouette Score | Cluster 1 Size | Cluster 2 Size |
|-----------|-----------|------------------|----------------|----------------|
| **Red Wine** | 2 | 0.205 | 542 (39.88%) | 817 (60.12%) |
| **White Wine** | 2 | 0.213 | 2,484 (62.71%) | 1,477 (37.29%) |

**Key Finding**: Both red and white wines naturally separate into 2 clusters that correlate with quality levels (higher vs. lower quality).

---

## 🚀 Key Features

### Feature Engineering

Created 4 domain-driven engineered features:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `acidity_ratio` | fixed_acidity / volatile_acidity | Balance between good and bad acids |
| `alcohol_va_ratio` | alcohol / volatile_acidity | Key quality indicator |
| `sugar_body_ratio` | residual_sugar / body | Sweetness balance |
| `sulfur_efficiency` | free_sulfur_dioxide efficiency | Preservative effectiveness |

### Critical Features for Quality

1. **Alcohol Content** (most important)
2. **Volatile Acidity** (strong negative correlation)
3. **Sulphates** (optimal levels indicate quality)
4. **Citric Acid** (freshness indicator)
5. **Engineered Ratios** (capture interactions)

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### [Model Comparison](docs/MODEL_COMPARISON.md)
- Detailed comparison of all supervised and unsupervised models
- Performance metrics and confusion matrices
- Strengths, weaknesses, and use cases for each model
- Supervised vs. unsupervised learning comparison
- Overall model ranking and recommendations

### [Conclusion](docs/CONCLUSION.md)
- Executive summary of achievements
- Project objectives and achievement status
- Practical applications for wine industry
- Technical achievements and challenges
- Quantitative and qualitative project impact

### [Discussion](docs/DISCUSSION.md)
- In-depth interpretation of results
- Model trade-offs and selection criteria
- Feature engineering impact analysis
- Class imbalance effects and solutions
- Supervised vs. unsupervised synergy
- Practical deployment considerations
- Limitations and threats to validity
- Future research directions
- Ethical considerations

---

## 💡 Key Insights & Recommendations

### Immediate Actions

1. **Restore XGBoost Model**: Highest accuracy (87.87%) makes it the best production candidate
2. **Create Model Ensemble**: Combine XGBoost (accuracy) + DNN (recall) for robust predictions
3. **Address Class Imbalance**: Implement SMOTE or class weighting for better minority class performance

### Long-term Strategy

4. **Maintain Model Diversity**: Keep interpretable (Decision Tree) + accurate (XGBoost) + complex (DNN) models
5. **Feature Engineering**: Continue domain-driven feature creation with wine experts
6. **Clustering Applications**: Use K-Means results for customer segmentation and marketing

---

## 🛠️ Technologies Used

### Programming Languages
- **Python 3.x**: Supervised and unsupervised learning implementations
- **R**: Original unsupervised clustering analysis

### Libraries & Frameworks
- **scikit-learn**: Clustering, preprocessing, metrics
- **PyTorch**: Deep neural network implementation
- **pandas/numpy**: Data processing
- **matplotlib/seaborn**: Visualization

---

## 🚦 How to Run

### Prerequisites
```bash
python >= 3.8
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

### Run Supervised Learning (DNN)
```bash
cd supervised_model/Neural_Network
python "Deep Neural Network.py"
```

### Run Unsupervised Learning
```bash
cd "unsupervised Cluster"
python unsupervised_analysis.py
```

### Generate Model Comparison
```bash
cd model_comparison
python comparison_analysis.py
```

---

## 👥 Team Contributions

### Supervised Learning Team
- Implemented 4 different supervised models
- Developed feature engineering pipeline
- Created Deep Neural Network (current active model)

### Unsupervised Learning Team
- Implemented K-Means clustering for red and white wines
- Conducted optimal k selection via silhouette analysis
- Generated cluster visualizations and heatmaps

### Model Comparison & Analysis
- Comprehensive model comparison across all approaches
- Generated comparative visualizations
- Documented findings, conclusion, and discussion

---

## 📝 Project Status

**Status**: ✅ **Complete** with recommendations for future enhancements

**Current State**:
- ✅ All models implemented and evaluated
- ✅ Comprehensive analysis and documentation completed
- ✅ Visualizations generated for all approaches
- ⚠️ Best models deleted (need restoration)
- ⚠️ Class imbalance not fully addressed

---

**Project Completion**: November 2025
**Course**: SYS 5170
**Topic**: Wine Quality Prediction using Machine Learning
