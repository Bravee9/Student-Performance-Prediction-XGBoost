# 📓 Jupyter Notebook Guide

## File: `brave9.ipynb`

Complete machine learning workflow with 35 cells organized in 10 main sections.

---

## 📋 Notebook Sections

### 1. Introduction (Cells 1-3)
- Project title and objectives
- Dataset overview  
- Navigation guide

### 2. Libraries (Cell 4)
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML algorithms
- `xgboost` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization
- All plots inline with `random_state=42`

### 3. Task Description (Cells 5-9)
- **Goal**: Predict math scores
- **Dataset**: 1,000 students, 8 features
- **Target**: Math score (0-100)
- **Factors**: Gender, race, parental education, lunch type, test prep

### 4. Data Loading (Cells 10-11)
```python
df = pd.read_csv("StudentsPerformance.csv")
```
- Loads 1,000 records
- Shows first 5 rows + data info
- **Result**: 0 missing values (clean data)

### 5. Exploratory Data Analysis (Cells 12-19)
- **Missing Values**: None ✓
- **Statistics**: Mean, std, quartiles
- **Distributions**: Histograms for all scores
- **Correlations**: Reading-Writing (0.954), Math-Reading (0.818)
- **Feature Relationships**: Boxplots by demographic groups
- **Key Finding**: SES (lunch) shows largest effect (10+ point gap)

### 6. Data Preprocessing (Cells 20-25)
- Separate target (y) and features (X)
- One-hot encoding for categorical variables
- 5 features → 11 features after encoding
- Train-test split: 80-20, random_state=42

### 7. Evaluation Function (Cell 26)
- Calculate RMSE, MAE, R² metrics
- Helper for model comparison

### 8. Linear Regression (Cells 27-29)
- **Purpose**: Baseline model
- **Results**: R²=0.23, RMSE=13.05, MAE=10.24
- **Interpretation**: Explains 23% of variance

### 9. XGBoost Regression (Cells 30-32)
- **Configuration**: 100 trees, max_depth=5, learning_rate=0.1
- **Results**: R²=0.26, RMSE=12.26, MAE=9.87
- **Improvement**: 13% better R², 6.1% better RMSE
- **Interpretation**: Explains 26% of variance

### 10. Model Comparison (Cell 33)
- Side-by-side metrics comparison
- Bar chart visualization
- **Winner**: XGBoost on all metrics

### 11. Feature Importance (Cell 34)
- Extract importance from XGBoost
- Top 5 predictors:
  1. lunch (34.2%)
  2. parental_education (21.5%)
  3. test_prep (18.9%)
  4. race/ethnicity (1.9%)
  5. gender (1.1%)
- Horizontal bar chart visualization

### 12. Conclusions (Cell 35)
- Summary of findings
- Policy recommendations (3 tiers)
- Limitations and future work

---

## 🔄 Execution Flow

**Important**: Run cells in order (1→35)
- Variables depend on previous cells
- Don't skip or reorder

**Runtime**: ~30-45 seconds for full notebook

**Output Types**:
- Console: Statistics, metrics
- Tables: DataFrames displayed
- Charts: 5+ visualizations (EDA, comparison, importance)
- Warnings: Safe to ignore (deprecations)

---

## 💾 Key Variables

### After Preprocessing (Cell 21)
- `X`: Features (1000 × 11)
- `y`: Target (1000,)
- `X_train`, `X_test`: Train/test split (800/200)
- `y_train`, `y_test`: Target split

### After Modeling (Cells 29, 32)
- `lr_model`: Linear Regression object
- `xgb_model`: XGBoost object
- `y_pred_lr`, `y_pred_xgb`: Predictions
- `lr_metrics`, `xgb_metrics`: Results dictionaries

### After Feature Importance (Cell 34)
- `feature_importance`: DataFrame with rankings

---

## 🎨 Visualizations Generated

1. **Histograms** (Cell 16) - Score distributions
2. **Heatmap** (Cell 17) - Correlation matrix
3. **Boxplots** (Cell 18) - Features vs math score
4. **Bar chart** (Cell 33) - Model comparison
5. **Bar chart** (Cell 34) - Feature importance

---

## 🛠️ Useful Code Snippets

### Get metrics
```python
print(f"XGBoost R²: {xgb_metrics['R2']:.4f}")
print(f"RMSE: {xgb_metrics['RMSE']:.2f}")
```

### Top features
```python
print(feature_importance.head(3))
```

### Make prediction
```python
new_data = X_test.iloc[[0]]
pred = xgb_model.predict(new_data)
```

### Save model
```python
import joblib
joblib.dump(xgb_model, 'model.pkl')
```

---

## ⚠️ Important Notes

**Data**:
- Never modify original CSV
- All transformations in notebook
- Safe to re-run anytime

**Reproducibility**:
- `random_state=42` everywhere
- Same results on re-runs
- Notebook is deterministic

**Dependencies**:
- Requires `requirements.txt` packages
- Python 3.8-3.10 recommended
- 2GB RAM minimum

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Run cell 4 again, check requirements |
| Data not loading | Verify CSV in SOURCE/ folder |
| Charts not showing | Run `%matplotlib inline` in cell 4 |
| Memory error | Restart kernel, check system RAM |


---

**Version**: 1.0  
**Last Updated**: November 2025  
**Status**: ✅ Production Ready  
**Author**: Bùi Quang Chiến (23001837)
