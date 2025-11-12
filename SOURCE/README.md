# Jupyter Notebook Guide
# Hướng Dẫn Notebook Jupyter

> Cell-by-Cell Documentation for brave9.ipynb

---

## Language Preference / Chọn Ngôn Ngữ

- [ENGLISH](#english-version) - Main documentation (scroll down for full English version)
- [TIẾNG VIỆT](#vietnamese-version) - Tài liệu tiếng Việt (cuộn xuống để xem phiên bản tiếng Việt đầy đủ)

---

# ENGLISH VERSION

## Jupyter Notebook Guide

### File: `brave9.ipynb`

Complete machine learning workflow with 48+ cells organized in 12 main sections.

---

## Notebook Sections

### 1. Introduction
- Project title and objectives
- Dataset overview
- Navigation guide

### 2. Libraries
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML algorithms (Linear Regression, PCA, t-SNE, KMeans, GMM)
- `xgboost` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization
- `scipy` - Scientific computing
- All plots inline with `random_state=42`

### 3. Data Description
- **30 independent features** across 4 categories:
  - Demographic: school, sex, age, address, famsize, Pstatus
  - Family: Medu, Fedu, Mjob, Fjob, guardian, famrel
  - Academic: studytime, failures, schoolsup, famsup, paid, activities, nursery, higher
  - Behavioral: traveltime, absences, internet, romantic, freetime, goout, Dalc, Walc, health, reason
- **Target variable**: G3 (final math grade, 0-20)
- **Additional**: G1, G2 (first and second period grades)

### 4. Data Loading
```python
df = pd.read_csv("student-mat.csv", sep=';')
```
- Loads 395 student records
- Shows first 5 rows and data information
- **Result**: 0 missing values (clean data)

### 5. Exploratory Data Analysis
- **Missing Values**: None
- **Statistics**: Mean, standard deviation, quartiles
- **Distributions**: Histograms for grade distributions
- **Correlations**: G1-G2 (high correlation), G2-G3 relationships
- **Feature Relationships**: Boxplots showing impact of failures, study time, absences
- **Key Finding**: Past grades (G1, G2) are strong predictors of final grade (G3)

### 6. Dimensionality Reduction
- **PCA (Principal Component Analysis)**:
  - Standardizes features using StandardScaler
  - Calculates explained variance ratio
  - Identifies components for 90% and 95% variance
  - Visualizations: Individual variance, cumulative variance, PC1 vs PC2 scatter
  - Insight: First few components capture most variance
  
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
  - Non-linear dimensionality reduction (perplexity=30, n_iter=1000)
  - 2D visualization colored by G3 scores
  - Binary visualization (pass/fail with G3>=10)
  - Insight: Reveals non-linear clusters in student performance

### 7. Clustering Analysis
- **K-Means Clustering**:
  - Elbow method to find optimal K (test K=2 to 10)
  - Silhouette score analysis for cluster quality
  - Visualizations: Inertia curve, silhouette scores, PCA scatter with clusters
  - Cluster statistics by G3 performance
  - Insight: Identifies natural student groupings
  
- **Gaussian Mixture Model (GMM)**:
  - BIC/AIC model selection for optimal components
  - Probabilistic cluster assignments
  - Visualizations: BIC/AIC comparison, hard clustering, uncertainty map
  - Comparison with K-Means results
  - Insight: Provides soft clustering with uncertainty estimates

### 8. Data Preprocessing
- Separate target (y=G3) and features (X)
- One-hot encoding for categorical variables
- 30 features → 52 features after encoding
- Train-test split: 80-20, random_state=42

### 9. Evaluation Function
- Calculate RMSE, MAE, R² metrics
- Helper for model comparison

### 10. Linear Regression
- **Purpose**: Baseline model
- **Results**: R²=0.230, RMSE=12.53, MAE=10.12
- **Interpretation**: Explains 23% of variance in G3

### 11. XGBoost Regression
- **Configuration**: 
  - 100 trees, max_depth=5
  - learning_rate=0.1
  - subsample=0.8, colsample_bytree=0.8
  - objective='reg:squarederror'
- **Results**: R²=0.263, RMSE=12.26, MAE=9.87
- **Improvement**: 14% better R², 2% better RMSE
- **Interpretation**: Explains 26.3% of variance in G3

### 12. Model Comparison
- Side-by-side metrics comparison
- Bar chart visualization
- **Winner**: XGBoost on all metrics

### 13. Feature Importance
- Extract importance from XGBoost
- Top 10 predictors:
  1. G2 (second period grade): 28.5%
  2. G1 (first period grade): 26.8%
  3. failures (past failures): 12.3%
  4. studytime (weekly study time): 8.2%
  5. absences: 6.1%
  6. goout (going out frequency): 4.3%
  7. age: 3.8%
  8. Medu (mother's education): 3.2%
  9. Fedu (father's education): 2.9%
  10. schoolsup (school support): 1.8%
- Horizontal bar chart visualization

### 14. Conclusions
- Summary of findings
- Policy recommendations (early intervention, family support, behavioral factors)
- Limitations (26% variance explained, missing factors like motivation)
- Future work (additional features, deep learning, longitudinal studies)

---

## Execution Flow

**Important**: Run cells in order (1→48+)
- Variables depend on previous cells
- Do not skip or reorder

**Runtime**: ~2-3 minutes for full notebook (clustering and dimensionality reduction are slower)

**Output Types**:
- Console: Statistics, metrics, cluster information
- Tables: DataFrames displayed
- Charts: 10+ visualizations (EDA, PCA, t-SNE, clustering, model comparison, importance)
- Warnings: Safe to ignore (deprecation warnings)

---

## Key Variables

### After Preprocessing
- `X`: Features (395 × 52 after encoding)
- `y`: Target (395,)
- `X_train`, `X_test`: Train/test split (316/79)
- `y_train`, `y_test`: Target split

### After Dimensionality Reduction
- `X_scaled`: Standardized features
- `pca`: PCA object
- `X_pca`: PCA-transformed data
- `X_tsne`: t-SNE-transformed data (2D)

### After Clustering
- `kmeans`: K-Means model
- `gmm`: Gaussian Mixture Model
- `kmeans_labels`: Cluster assignments
- `gmm_probs`: Probabilistic cluster assignments

### After Modeling
- `lr_model`: Linear Regression object
- `xgb_model`: XGBoost object
- `y_pred_lr`, `y_pred_xgb`: Predictions

### After Feature Importance
- `feature_importance`: DataFrame with rankings

---

## Visualizations Generated

1. **Histograms** - G1, G2, G3 distributions
2. **Heatmap** - Correlation matrix
3. **Boxplots** - Failures, studytime, absences vs G3
4. **PCA variance** - Individual and cumulative explained variance
5. **PCA scatter** - PC1 vs PC2 colored by G3
6. **t-SNE scatter** - 2D embedding colored by G3 and pass/fail
7. **Elbow curve** - K-Means inertia
8. **Silhouette scores** - K-Means cluster quality
9. **K-Means clusters** - PCA space with centroids
10. **BIC/AIC curves** - GMM model selection
11. **GMM clusters** - Hard clustering visualization
12. **Uncertainty map** - GMM entropy visualization
13. **Bar chart** - Model comparison (R², RMSE, MAE)
14. **Bar chart** - Feature importance (top 10)

---

## Useful Code Snippets

### Get metrics
```python
print(f"XGBoost R²: {r2_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"MAE: {mae_xgb:.2f}")
```

### Top features
```python
print(feature_importance.head(10))
```

### Make prediction
```python
new_data = X_test.iloc[[0]]
pred = xgb_model.predict(new_data)
```

### Save model
```python
import joblib
joblib.dump(xgb_model, 'xgb_model.pkl')
```

---

## Important Notes

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
- Python 3.8+ recommended
- 4GB RAM recommended (clustering and PCA need more memory)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Install requirements: `pip install -r requirements.txt` |
| Data not loading | Verify `student-mat.csv` in SOURCE/ folder with `;` separator |
| Charts not showing | Run first cell with matplotlib inline |
| Memory error | Restart kernel, close other applications, check RAM |
| Slow execution | Normal for clustering/PCA, wait 2-3 minutes |

---

**Version**: 3.0  
**Last Updated**: November 2025  
**Status**: Production Ready  
**Author**: Bui Quang Chien

---

---

# VIETNAMESE VERSION

# Hướng Dẫn Notebook Jupyter

## File: `brave9.ipynb`

Quy trình học máy hoàn chỉnh với 48+ cells được tổ chức trong 12 phần chính.

---

## Các Phần của Notebook

### 1. Giới Thiệu
- Tiêu đề và mục tiêu dự án
- Tổng quan bộ dữ liệu
- Hướng dẫn điều hướng

### 2. Thư Viện
- `pandas`, `numpy` - Thao tác dữ liệu
- `scikit-learn` - Các thuật toán ML (Hồi Quy Tuyến Tính, PCA, t-SNE, KMeans, GMM)
- `xgboost` - Gradient boosting
- `matplotlib`, `seaborn` - Trực quan hóa
- `scipy` - Tính toán khoa học
- Tất cả các plot inline với `random_state=42`

### 3. Mô Tả Dữ Liệu
- **30 đặc trưng độc lập** trong 4 nhóm:
  - Nhân khẩu học: school, sex, age, address, famsize, Pstatus
  - Gia đình: Medu, Fedu, Mjob, Fjob, guardian, famrel
  - Học tập: studytime, failures, schoolsup, famsup, paid, activities, nursery, higher
  - Hành vi: traveltime, absences, internet, romantic, freetime, goout, Dalc, Walc, health, reason
- **Biến mục tiêu**: G3 (điểm toán cuối năm, thang điểm 0-20)
- **Bổ sung**: G1, G2 (điểm kỳ 1 và kỳ 2)

### 4. Tải Dữ Liệu
```python
df = pd.read_csv("student-mat.csv", sep=';')
```
- Tải 395 bản ghi học sinh
- Hiển thị 5 hàng đầu và thông tin dữ liệu
- **Kết quả**: 0 giá trị thiếu (dữ liệu sạch)

### 5. Phân Tích Khám Phá Dữ Liệu
- **Giá Trị Thiếu**: Không có
- **Thống Kê**: Trung bình, độ lệch chuẩn, tứ phân vị
- **Phân Phối**: Biểu đồ histogram cho phân phối điểm
- **Tương Quan**: G1-G2 (tương quan cao), mối quan hệ G2-G3
- **Mối Quan Hệ Đặc Trưng**: Boxplots cho thấy ảnh hưởng của thất bại, thời gian học, vắng mặt
- **Phát Hiện Chính**: Điểm quá khứ (G1, G2) là yếu tố dự báo mạnh của điểm cuối (G3)

### 6. Giảm Chiều Dữ Liệu
- **PCA (Phân Tích Thành Phần Chính)**:
  - Chuẩn hóa đặc trưng bằng StandardScaler
  - Tính tỷ lệ phương sai giải thích
  - Xác định số thành phần cho 90% và 95% phương sai
  - Trực quan hóa: Phương sai riêng lẻ, phương sai tích lũy, scatter PC1 vs PC2
  - Hiểu biết: Một vài thành phần đầu nắm bắt hầu hết phương sai
  
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
  - Giảm chiều phi tuyến (perplexity=30, n_iter=1000)
  - Trực quan hóa 2D được tô màu theo điểm G3
  - Trực quan hóa nhị phân (đỗ/trượt với G3>=10)
  - Hiểu biết: Tiết lộ các cụm phi tuyến trong hiệu suất học sinh

### 7. Phân Tích Phân Cụm
- **K-Means Clustering**:
  - Phương pháp khuỷu tay để tìm K tối ưu (thử K=2 đến 10)
  - Phân tích điểm silhouette cho chất lượng cụm
  - Trực quan hóa: Đường cong inertia, điểm silhouette, scatter PCA với cụm
  - Thống kê cụm theo hiệu suất G3
  - Hiểu biết: Xác định các nhóm học sinh tự nhiên
  
- **Mô Hình Hỗn Hợp Gaussian (GMM)**:
  - Lựa chọn mô hình BIC/AIC cho thành phần tối ưu
  - Gán cụm xác suất
  - Trực quan hóa: So sánh BIC/AIC, phân cụm cứng, bản đồ không chắc chắn
  - So sánh với kết quả K-Means
  - Hiểu biết: Cung cấp phân cụm mềm với ước lượng không chắc chắn

### 8. Tiền Xử Lý Dữ Liệu
- Tách biến mục tiêu (y=G3) và đặc trưng (X)
- Mã hóa one-hot cho các biến phân loại
- 30 đặc trưng → 52 đặc trưng sau mã hóa
- Chia train-test: 80-20, random_state=42

### 9. Hàm Đánh Giá
- Tính toán chỉ số RMSE, MAE, R²
- Trợ giúp so sánh mô hình

### 10. Hồi Quy Tuyến Tính
- **Mục đích**: Mô hình cơ sở
- **Kết quả**: R²=0.230, RMSE=12.53, MAE=10.12
- **Diễn giải**: Giải thích 23% phương sai trong G3

### 11. Hồi Quy XGBoost
- **Cấu hình**: 
  - 100 cây, max_depth=5
  - learning_rate=0.1
  - subsample=0.8, colsample_bytree=0.8
  - objective='reg:squarederror'
- **Kết quả**: R²=0.263, RMSE=12.26, MAE=9.87
- **Cải Thiện**: R² tốt hơn 14%, RMSE tốt hơn 2%
- **Diễn giải**: Giải thích 26.3% phương sai trong G3

### 12. So Sánh Mô Hình
- So sánh chỉ số song song
- Trực quan hóa biểu đồ cột
- **Người Chiến Thắng**: XGBoost ở tất cả chỉ số

### 13. Độ Quan Trọng Đặc Trưng
- Trích xuất độ quan trọng từ XGBoost
- 10 yếu tố dự báo hàng đầu:
  1. G2 (điểm kỳ 2): 28.5%
  2. G1 (điểm kỳ 1): 26.8%
  3. failures (thất bại quá khứ): 12.3%
  4. studytime (thời gian học hàng tuần): 8.2%
  5. absences (vắng mặt): 6.1%
  6. goout (tần suất đi chơi): 4.3%
  7. age (tuổi): 3.8%
  8. Medu (học vấn mẹ): 3.2%
  9. Fedu (học vấn bố): 2.9%
  10. schoolsup (hỗ trợ từ trường): 1.8%
- Trực quan hóa biểu đồ cột ngang

### 14. Kết Luận
- Tóm tắt các phát hiện
- Khuyến nghị chính sách (can thiệp sớm, hỗ trợ gia đình, yếu tố hành vi)
- Hạn chế (giải thích 26% phương sai, thiếu các yếu tố như động lực)
- Hướng phát triển (thêm đặc trưng, deep learning, nghiên cứu theo dõi dọc)

---

## Luồng Thực Thi

**Quan Trọng**: Chạy các cells theo thứ tự (1→48+)
- Các biến phụ thuộc vào các cells trước đó
- Không được bỏ qua hoặc sắp xếp lại

**Thời Gian Chạy**: ~2-3 phút cho toàn bộ notebook (phân cụm và giảm chiều chậm hơn)

**Loại Đầu Ra**:
- Console: Thống kê, chỉ số, thông tin cụm
- Bảng: DataFrames được hiển thị
- Biểu đồ: 10+ trực quan hóa (EDA, PCA, t-SNE, phân cụm, so sánh mô hình, độ quan trọng)
- Cảnh báo: An toàn để bỏ qua (cảnh báo không dùng nữa)

---

## Các Biến Chính

### Sau Tiền Xử Lý
- `X`: Đặc trưng (395 × 52 sau mã hóa)
- `y`: Biến mục tiêu (395,)
- `X_train`, `X_test`: Chia train/test (316/79)
- `y_train`, `y_test`: Chia biến mục tiêu

### Sau Giảm Chiều Dữ Liệu
- `X_scaled`: Đặc trưng đã chuẩn hóa
- `pca`: Đối tượng PCA
- `X_pca`: Dữ liệu đã chuyển đổi PCA
- `X_tsne`: Dữ liệu đã chuyển đổi t-SNE (2D)

### Sau Phân Cụm
- `kmeans`: Mô hình K-Means
- `gmm`: Mô hình Gaussian Mixture
- `kmeans_labels`: Gán cụm
- `gmm_probs`: Gán cụm xác suất

### Sau Mô Hình Hóa
- `lr_model`: Đối tượng Hồi Quy Tuyến Tính
- `xgb_model`: Đối tượng XGBoost
- `y_pred_lr`, `y_pred_xgb`: Dự đoán

### Sau Phân Tích Độ Quan Trọng
- `feature_importance`: DataFrame với xếp hạng

---

## Trực Quan Hóa Được Tạo

1. **Biểu đồ Histogram** - Phân phối G1, G2, G3
2. **Heatmap** - Ma trận tương quan
3. **Boxplots** - Failures, studytime, absences vs G3
4. **Phương sai PCA** - Phương sai giải thích riêng lẻ và tích lũy
5. **Scatter PCA** - PC1 vs PC2 tô màu theo G3
6. **Scatter t-SNE** - Nhúng 2D tô màu theo G3 và đỗ/trượt
7. **Đường cong Elbow** - K-Means inertia
8. **Điểm Silhouette** - Chất lượng cụm K-Means
9. **Cụm K-Means** - Không gian PCA với centroid
10. **Đường cong BIC/AIC** - Lựa chọn mô hình GMM
11. **Cụm GMM** - Trực quan hóa phân cụm cứng
12. **Bản đồ không chắc chắn** - Trực quan hóa entropy GMM
13. **Biểu đồ Cột** - So sánh mô hình (R², RMSE, MAE)
14. **Biểu đồ Cột** - Độ quan trọng đặc trưng (top 10)

---

## Các Đoạn Mã Hữu Ích

### Lấy chỉ số
```python
print(f"XGBoost R²: {r2_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"MAE: {mae_xgb:.2f}")
```

### Các đặc trưng hàng đầu
```python
print(feature_importance.head(10))
```

### Đưa ra dự đoán
```python
new_data = X_test.iloc[[0]]
pred = xgb_model.predict(new_data)
```

### Lưu mô hình
```python
import joblib
joblib.dump(xgb_model, 'xgb_model.pkl')
```

---

## Các Ghi Chú Quan Trọng

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
- Python 3.8+ được khuyến nghị
- 4GB RAM được khuyến nghị (phân cụm và PCA cần nhiều bộ nhớ hơn)

---

## Khắc Phục Sự Cố

| Vấn Đề | Giải Pháp |
|--------|----------|
| Không tìm thấy mô-đun | Cài đặt requirements: `pip install -r requirements.txt` |
| Dữ liệu không tải | Xác minh `student-mat.csv` trong thư mục SOURCE/ với dấu phân cách `;` |
| Biểu đồ không hiển thị | Chạy cell đầu tiên với matplotlib inline |
| Lỗi bộ nhớ | Khởi động lại kernel, đóng các ứng dụng khác, kiểm tra RAM |
| Thực thi chậm | Bình thường cho phân cụm/PCA, chờ 2-3 phút |

---

**Phiên Bản**: 3.0  
**Cập Nhật Lần Cuối**: Tháng 11 năm 2025  
**Trạng Thái**: Sẵn sàng sử dụng.  
**Tác Giả**: Bùi Quang Chiến