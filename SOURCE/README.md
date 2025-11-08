# 📓 Jupyter Notebook Guide

## File: `brave9.ipynb`

Complete machine learning workflow with 37 cells organized in 10 main sections.

Quy trình học máy hoàn chỉnh với 37 cells được tổ chức trong 10 phần chính.

---

## 📋 Notebook Sections / Các Phần của Notebook

### 1. Introduction (Cells 1-3) / Giới Thiệu (Cells 1-3)
- Project title and objectives / Tiêu đề và mục tiêu dự án
- Dataset overview / Tổng quan bộ dữ liệu
- Navigation guide / Hướng dẫn điều hướng

### 2. Libraries (Cell 4) / Thư Viện (Cell 4)
- `pandas`, `numpy` - Data manipulation / Thao tác dữ liệu
- `scikit-learn` - ML algorithms / Các thuật toán ML
- `xgboost` - Gradient boosting / Gradient boosting
- `matplotlib`, `seaborn` - Visualization / Trực quan hóa
- All plots inline with `random_state=42` / Tất cả các plot inline với `random_state=42`

### 3. Task Description (Cells 5-9) / Mô Tả Nhiệm Vụ (Cells 5-9)
- **Goal / Mục tiêu**: Predict math scores / Dự đoán điểm toán
- **Dataset / Bộ dữ liệu**: 1,000 students / học sinh, 8 features / đặc trưng
- **Target / Biến mục tiêu**: Math score / Điểm toán (0-100)
- **Factors / Yếu tố**: Gender / giới tính, race / chủng tộc, parental education / trình độ cha mẹ, lunch type / loại bữa trưa, test prep / luyện thi

### 4. Data Loading (Cells 10-11) / Tải Dữ Liệu (Cells 10-11)
```python
df = pd.read_csv("StudentsPerformance.csv")
```
- Loads 1,000 records / Tải 1.000 bản ghi
- Shows first 5 rows + data info / Hiển thị 5 hàng đầu + thông tin dữ liệu
- **Result / Kết quả**: 0 missing values (clean data) / 0 giá trị thiếu (dữ liệu sạch)

### 5. Exploratory Data Analysis (Cells 12-19) / Phân Tích Khám Phá Dữ Liệu (Cells 12-19)
- **Missing Values / Giá Trị Thiếu**: None ✓
- **Statistics / Thống Kê**: Mean, std, quartiles / Trung bình, độ lệch chuẩn, tứ phân vị
- **Distributions / Phân Phối**: Histograms for all scores / Biểu đồ cho tất cả các điểm
- **Correlations / Tương Quan**: Reading-Writing (0.954), Math-Reading (0.818) / Đọc-Viết (0.954), Toán-Đọc (0.818)
- **Feature Relationships / Mối Quan Hệ Đặc Trưng**: Boxplots by demographic groups / Boxplots theo nhóm nhân khẩu học
- **Key Finding / Phát Hiện Chính**: SES (lunch) shows largest effect (10+ point gap) / KXH (bữa trưa) cho thấy ảnh hưởng lớn nhất (chênh lệch 10+ điểm)

### 6. Data Preprocessing (Cells 20-25) / Tiền Xử Lý Dữ Liệu (Cells 20-25)
- Separate target (y) and features (X) / Tách biến mục tiêu (y) và đặc trưng (X)
- One-hot encoding for categorical variables / Mã hóa one-hot cho các biến phân loại
- 5 features → 11 features after encoding / 5 đặc trưng → 11 đặc trưng sau mã hóa
- Train-test split: 80-20, random_state=42 / Chia train-test: 80-20, random_state=42

### 7. Evaluation Function (Cell 26) / Hàm Đánh Giá (Cell 26)
- Calculate RMSE, MAE, R² metrics / Tính toán chỉ số RMSE, MAE, R²
- Helper for model comparison / Trợ giúp so sánh mô hình

### 8. Linear Regression (Cells 27-29) / Hồi Quy Tuyến Tính (Cells 27-29)
- **Purpose / Mục đích**: Baseline model / Mô hình cơ sở
- **Results / Kết quả**: R²=0.23, RMSE=13.05, MAE=10.24
- **Interpretation / Diễn giải**: Explains 23% of variance / Giải thích 23% phương sai

### 9. XGBoost Regression (Cells 30-32) / Hồi Quy XGBoost (Cells 30-32)
- **Configuration / Cấu hình**: 100 trees, max_depth=5, learning_rate=0.1
- **Results / Kết quả**: R²=0.26, RMSE=12.26, MAE=9.87
- **Improvement / Cải Thiện**: 13% better R², 6.1% better RMSE / R² tốt hơn 13%, RMSE tốt hơn 6,1%
- **Interpretation / Diễn giải**: Explains 26% of variance / Giải thích 26% phương sai

### 10. Model Comparison (Cell 33) / So Sánh Mô Hình (Cell 33)
- Side-by-side metrics comparison / So sánh chỉ số song song
- Bar chart visualization / Trực quan hóa biểu đồ cột
- **Winner / Người Chiến Thắng**: XGBoost on all metrics / XGBoost ở tất cả chỉ số

### 11. Feature Importance (Cell 34) / Độ Quan Trọng Đặc Trưng (Cell 34)
- Extract importance from XGBoost / Trích xuất độ quan trọng từ XGBoost
- Top 5 predictors / 5 yếu tố dự báo hàng đầu:
  1. lunch (34.2%)
  2. parental_education (21.5%)
  3. test_prep (18.9%)
  4. race/ethnicity (1.9%)
  5. gender (1.1%)
- Horizontal bar chart visualization / Trực quan hóa biểu đồ cột ngang

### 12. Conclusions (Cell 37) / Kết Luận (Cell 37)
- Summary of findings / Tóm tắt các phát hiện
- Policy recommendations (3 tiers) / Khuyến nghị chính sách (3 cấp)
- Limitations and future work / Hạn chế và hướng phát triển tương lai

---

## 🔄 Execution Flow / Luồng Thực Thi

**Important / Quan Trọng**: Run cells in order (1→37) / Chạy các cells theo thứ tự (1→37)
- Variables depend on previous cells / Các biến phụ thuộc vào các cells trước đó
- Don't skip or reorder / Không được bỏ qua hoặc sắp xếp lại

**Runtime / Thời Gian Chạy**: ~30-45 seconds for full notebook / ~30-45 giây cho toàn bộ notebook

**Output Types / Loại Đầu Ra**:
- Console: Statistics, metrics / Thống kê, chỉ số
- Tables: DataFrames displayed / Bảng: DataFrames được hiển thị
- Charts: 5+ visualizations (EDA, comparison, importance) / Biểu đồ: 5+ trực quan hóa (EDA, so sánh, độ quan trọng)
- Warnings: Safe to ignore (deprecations) / Cảnh báo: An toàn để bỏ qua (không dùng nữa)

---

## 💾 Key Variables / Các Biến Chính

### After Preprocessing (Cell 21) / Sau Tiền Xử Lý (Cell 21)
- `X`: Features / Đặc trưng (1000 × 11)
- `y`: Target / Biến mục tiêu (1000,)
- `X_train`, `X_test`: Train/test split / Chia train/test (800/200)
- `y_train`, `y_test`: Target split / Chia biến mục tiêu

### After Modeling (Cells 29, 32) / Sau Mô Hình Hóa (Cells 29, 32)
- `lr_model`: Linear Regression object / Đối tượng Hồi Quy Tuyến Tính
- `xgb_model`: XGBoost object / Đối tượng XGBoost
- `y_pred_lr`, `y_pred_xgb`: Predictions / Dự đoán
- `lr_metrics`, `xgb_metrics`: Results dictionaries / Từ điển kết quả

### After Feature Importance (Cell 34) / Sau Phân Tích Độ Quan Trọng (Cell 34)
- `feature_importance`: DataFrame with rankings / DataFrame với xếp hạng

---

## 🎨 Visualizations Generated / Trực Quan Hóa Được Tạo

1. **Histograms** (Cell 16) - Score distributions / Phân phối điểm
2. **Heatmap** (Cell 17) - Correlation matrix / Ma trận tương quan
3. **Boxplots** (Cell 18) - Features vs math score / Đặc trưng vs điểm toán
4. **Bar chart** (Cell 33) - Model comparison / So sánh mô hình
5. **Bar chart** (Cell 34) - Feature importance / Độ quan trọng đặc trưng

---

## 🛠️ Useful Code Snippets / Các Đoạn Mã Hữu Ích

### Get metrics / Lấy chỉ số
```python
print(f"XGBoost R²: {xgb_metrics['R2']:.4f}")
print(f"RMSE: {xgb_metrics['RMSE']:.2f}")
```

### Top features / Các đặc trưng hàng đầu
```python
print(feature_importance.head(3))
```

### Make prediction / Đưa ra dự đoán
```python
new_data = X_test.iloc[[0]]
pred = xgb_model.predict(new_data)
```

### Save model / Lưu mô hình
```python
import joblib
joblib.dump(xgb_model, 'model.pkl')
```

---

## ⚠️ Important Notes / Các Ghi Chú Quan Trọng

**Data / Dữ Liệu**:
- Never modify original CSV / Không bao giờ sửa đổi CSV gốc
- All transformations in notebook / Tất cả các biến đổi trong notebook
- Safe to re-run anytime / An toàn để chạy lại bất kỳ lúc nào

**Reproducibility / Tái Lập**:
- `random_state=42` everywhere / `random_state=42` ở mọi nơi
- Same results on re-runs / Kết quả giống nhau khi chạy lại
- Notebook is deterministic / Notebook là xác định

**Dependencies / Thư Viện Phụ Thuộc**:
- Requires `requirements.txt` packages / Yêu cầu các gói trong `requirements.txt`
- Python 3.8-3.10 recommended / Python 3.8-3.10 được khuyến cáo
- 2GB RAM minimum / Tối thiểu 2GB RAM

---

## ❓ Troubleshooting / Khắc Phục Sự Cố

| Problem / Vấn Đề | Solution / Giải Pháp |
|--------|----------|
| Module not found / Không tìm thấy mô-đun | Run cell 4 again, check requirements / Chạy cell 4 lại, kiểm tra requirements |
| Data not loading / Dữ liệu không tải | Verify CSV in SOURCE/ folder / Xác minh CSV trong thư mục SOURCE/ |
| Charts not showing / Biểu đồ không hiển thị | Run `%matplotlib inline` in cell 4 / Chạy `%matplotlib inline` trong cell 4 |
| Memory error / Lỗi bộ nhớ | Restart kernel, check system RAM / Khởi động lại kernel, kiểm tra RAM hệ thống |

---

**Version / Phiên Bản**: 1.0  
**Last Updated / Cập Nhật Lần Cuối**: November 2025 / Tháng 11 năm 2025  
**Status / Trạng Thái**: ✅ Production Ready / Sẵn Sàng Sản Xuất  
**Author / Tác Giả**: Bùi Quang Chiến
