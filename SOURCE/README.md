# 📓 Jupyter Notebook Guide
# Hướng Dẫn Notebook Jupyter

> **Cell-by-Cell Documentation for brave9.ipynb**

---

## 📖 Language Preference / Chọn Ngôn Ngữ

- **[ENGLISH](#english-version)** - Main documentation (scroll down for full English version)
- **[TIẾNG VIỆT](#vietnamese-version)** - Tài liệu tiếng Việt (cuộn xuống để xem phiên bản tiếng Việt đầy đủ)

---

# ENGLISH VERSION

## 📓 Jupyter Notebook Guide

### File: `brave9.ipynb`

Complete machine learning workflow with 37 cells organized in 10 main sections.

---

## 📋 Notebook Sections

## 📋 Notebook Sections

### 1. Introduction
- Project title and objectives
- Dataset overview
- Navigation guide

### 2. Libraries
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML algorithms
- `xgboost` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization
- All plots inline with `random_state=42`

### 3. Task Description
- **Goal**: Predict math scores
- **Dataset**: 1,000 students, 8 features
- **Target**: Math score (0-100)
- **Factors**: Gender, race, parental education, lunch type, test prep

### 4. Data Loading
```python
df = pd.read_csv("StudentsPerformance.csv")
```
- Loads 1,000 records
- Shows first 5 rows + data info
- **Result**: 0 missing values (clean data)

### 5. Exploratory Data Analysis
- **Missing Values**: None ✓
- **Statistics**: Mean, std, quartiles
- **Distributions**: Histograms for all scores
- **Correlations**: Reading-Writing (0.954), Math-Reading (0.818)
- **Feature Relationships**: Boxplots by demographic groups
- **Key Finding**: SES (lunch) shows largest effect (10+ point gap)

### 6. Data Preprocessing
- Separate target (y) and features (X)
- One-hot encoding for categorical variables
- 5 features → 11 features after encoding
- Train-test split: 80-20, random_state=42

### 7. Evaluation Function
- Calculate RMSE, MAE, R² metrics
- Helper for model comparison

### 8. Linear Regression
- **Purpose**: Baseline model
- **Results**: R²=0.23, RMSE=13.05, MAE=10.24
- **Interpretation**: Explains 23% of variance

### 9. XGBoost Regression
- **Configuration**: 100 trees, max_depth=5, learning_rate=0.1
- **Results**: R²=0.26, RMSE=12.26, MAE=9.87
- **Improvement**: 13% better R², 6.1% better RMSE
- **Interpretation**: Explains 26% of variance

### 10. Model Comparison
- Side-by-side metrics comparison
- Bar chart visualization
- **Winner**: XGBoost on all metrics

### 11. Feature Importance
- Extract importance from XGBoost
- Top 5 predictors:
  1. lunch (34.2%)
  2. parental_education (21.5%)
  3. test_prep (18.9%)
  4. race/ethnicity (1.9%)
  5. gender (1.1%)
- Horizontal bar chart visualization

### 12. Conclusions
- Summary of findings
- Policy recommendations (3 tiers)
- Limitations and future work

---

## 🔄 Execution Flow

**Important**: Run cells in order (1→37)
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

### After Preprocessing
- `X`: Features (1000 × 11)
- `y`: Target (1000,)
- `X_train`, `X_test`: Train/test split (800/200)
- `y_train`, `y_test`: Target split

### After Modeling
- `lr_model`: Linear Regression object
- `xgb_model`: XGBoost object
- `y_pred_lr`, `y_pred_xgb`: Predictions
- `lr_metrics`, `xgb_metrics`: Results dictionaries

### After Feature Importance
- `feature_importance`: DataFrame with rankings

---

## 🎨 Visualizations Generated

1. **Histograms** - Score distributions
2. **Heatmap** - Correlation matrix
3. **Boxplots** - Features vs math score
4. **Bar chart** - Model comparison
5. **Bar chart** - Feature importance

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
**Author**: Bùi Quang Chiến

---

---

# VIETNAMESE VERSION

# 📓 Hướng Dẫn Notebook Jupyter

## File: `brave9.ipynb`

Quy trình học máy hoàn chỉnh với 37 cells được tổ chức trong 10 phần chính.

---

## 📋 Các Phần của Notebook

### 1. Giới Thiệu (Cells 1-3)
- Tiêu đề và mục tiêu dự án
- Tổng quan bộ dữ liệu
- Hướng dẫn điều hướng

### 2. Thư Viện (Cell 4)
- `pandas`, `numpy` - Thao tác dữ liệu
- `scikit-learn` - Các thuật toán ML
- `xgboost` - Gradient boosting
- `matplotlib`, `seaborn` - Trực quan hóa
- Tất cả các plot inline với `random_state=42`

### 3. Mô Tả Nhiệm Vụ (Cells 5-9)
- **Mục tiêu**: Dự đoán điểm toán
- **Bộ dữ liệu**: 1.000 học sinh, 8 đặc trưng
- **Biến mục tiêu**: Điểm toán (0-100)
- **Yếu tố**: Giới tính, chủng tộc, trình độ cha mẹ, loại bữa trưa, luyện thi

### 4. Tải Dữ Liệu (Cells 10-11)
```python
df = pd.read_csv("StudentsPerformance.csv")
```
- Tải 1.000 bản ghi
- Hiển thị 5 hàng đầu + thông tin dữ liệu
- **Kết quả**: 0 giá trị thiếu (dữ liệu sạch)

### 5. Phân Tích Khám Phá Dữ Liệu (Cells 12-19)
- **Giá Trị Thiếu**: Không có ✓
- **Thống Kê**: Trung bình, độ lệch chuẩn, tứ phân vị
- **Phân Phối**: Biểu đồ cho tất cả các điểm
- **Tương Quan**: Đọc-Viết (0.954), Toán-Đọc (0.818)
- **Mối Quan Hệ Đặc Trưng**: Boxplots theo nhóm nhân khẩu học
- **Phát Hiện Chính**: KXH (bữa trưa) cho thấy ảnh hưởng lớn nhất (chênh lệch 10+ điểm)

### 6. Tiền Xử Lý Dữ Liệu (Cells 20-25)
- Tách biến mục tiêu (y) và đặc trưng (X)
- Mã hóa one-hot cho các biến phân loại
- 5 đặc trưng → 11 đặc trưng sau mã hóa
- Chia train-test: 80-20, random_state=42

### 7. Hàm Đánh Giá (Cell 26)
- Tính toán chỉ số RMSE, MAE, R²
- Trợ giúp so sánh mô hình

### 8. Hồi Quy Tuyến Tính (Cells 27-29)
- **Mục đích**: Mô hình cơ sở
- **Kết quả**: R²=0.23, RMSE=13.05, MAE=10.24
- **Diễn giải**: Giải thích 23% phương sai

### 9. Hồi Quy XGBoost (Cells 30-32)
- **Cấu hình**: 100 cây, max_depth=5, learning_rate=0.1
- **Kết quả**: R²=0.26, RMSE=12.26, MAE=9.87
- **Cải Thiện**: R² tốt hơn 13%, RMSE tốt hơn 6,1%
- **Diễn giải**: Giải thích 26% phương sai

### 10. So Sánh Mô Hình (Cell 33)
- So sánh chỉ số song song
- Trực quan hóa biểu đồ cột
- **Người Chiến Thắng**: XGBoost ở tất cả chỉ số

### 11. Độ Quan Trọng Đặc Trưng (Cell 34)
- Trích xuất độ quan trọng từ XGBoost
- 5 yếu tố dự báo hàng đầu:
  1. lunch (34.2%)
  2. parental_education (21.5%)
  3. test_prep (18.9%)
  4. race/ethnicity (1.9%)
  5. gender (1.1%)
- Trực quan hóa biểu đồ cột ngang

### 12. Kết Luận (Cell 37)
- Tóm tắt các phát hiện
- Khuyến nghị chính sách (3 cấp)
- Hạn chế và hướng phát triển tương lai

---

## 🔄 Luồng Thực Thi

**Quan Trọng**: Chạy các cells theo thứ tự (1→37)
- Các biến phụ thuộc vào các cells trước đó
- Không được bỏ qua hoặc sắp xếp lại

**Thời Gian Chạy**: ~30-45 giây cho toàn bộ notebook

**Loại Đầu Ra**:
- Console: Thống kê, chỉ số
- Bảng: DataFrames được hiển thị
- Biểu đồ: 5+ trực quan hóa (EDA, so sánh, độ quan trọng)
- Cảnh báo: An toàn để bỏ qua (không dùng nữa)

---

## 💾 Các Biến Chính

### Sau Tiền Xử Lý
- `X`: Đặc trưng (1000 × 11)
- `y`: Biến mục tiêu (1000,)
- `X_train`, `X_test`: Chia train/test (800/200)
- `y_train`, `y_test`: Chia biến mục tiêu

### Sau Mô Hình Hóa
- `lr_model`: Đối tượng Hồi Quy Tuyến Tính
- `xgb_model`: Đối tượng XGBoost
- `y_pred_lr`, `y_pred_xgb`: Dự đoán
- `lr_metrics`, `xgb_metrics`: Từ điển kết quả

### Sau Phân Tích Độ Quan Trọng
- `feature_importance`: DataFrame với xếp hạng

---

## 🎨 Trực Quan Hóa Được Tạo

1. **Biểu đồ Histogram** - Phân phối điểm
2. **Heatmap** - Ma trận tương quan
3. **Boxplots** - Đặc trưng vs điểm toán
4. **Biểu đồ Cột** - So sánh mô hình
5. **Biểu đồ Cột** - Độ quan trọng đặc trưng

---

## 🛠️ Các Đoạn Mã Hữu Ích

### Lấy chỉ số
```python
print(f"XGBoost R²: {xgb_metrics['R2']:.4f}")
print(f"RMSE: {xgb_metrics['RMSE']:.2f}")
```

### Các đặc trưng hàng đầu
```python
print(feature_importance.head(3))
```

### Đưa ra dự đoán
```python
new_data = X_test.iloc[[0]]
pred = xgb_model.predict(new_data)
```

### Lưu mô hình
```python
import joblib
joblib.dump(xgb_model, 'model.pkl')
```

---

## ⚠️ Các Ghi Chú Quan Trọng

**Dữ Liệu**:
- Không bao giờ sửa đổi CSV gốc
- Tất cả các biến đổi trong notebook
- An toàn để chạy lại bất kỳ lúc nào

**Tái Lập**:
- `random_state=42` ở mọi nơi
- Kết quả giống nhau khi chạy lại
- Notebook là xác định

**Thư Viện Phụ Thuộc**:
- Yêu cầu các gói trong `requirements.txt`
- Python 3.8-3.10 được khuyến cáo
- Tối thiểu 2GB RAM

---

## ❓ Khắc Phục Sự Cố

| Vấn Đề | Giải Pháp |
|--------|----------|
| Không tìm thấy mô-đun | Chạy cell 4 lại, kiểm tra requirements |
| Dữ liệu không tải | Xác minh CSV trong thư mục SOURCE/ |
| Biểu đồ không hiển thị | Chạy `%matplotlib inline` trong cell 4 |
| Lỗi bộ nhớ | Khởi động lại kernel, kiểm tra RAM hệ thống |

---

**Phiên Bản**: 1.0  
**Cập Nhật Lần Cuối**: Tháng 11 năm 2025  
**Trạng Thái**: ✅ Sẵn Sàng Sản Xuất  
**Tác Giả**: Bùi Quang Chiến