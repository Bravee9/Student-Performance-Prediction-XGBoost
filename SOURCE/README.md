# 📓 Hướng Dẫn Jupyter Notebook

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
- **Giá Trị Thiếu**: Không ✓
- **Thống Kê**: Trung bình, độ lệch chuẩn, tứ phân vị
- **Phân Phối**: Biểu đồ cho tất cả các điểm
- **Tương Quan**: Đọc-Viết (0,954), Toán-Đọc (0,818)
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
- **Kết quả**: R²=0,23, RMSE=13,05, MAE=10,24
- **Diễn giải**: Giải thích 23% phương sai

### 9. Hồi Quy XGBoost (Cells 30-32)
- **Cấu hình**: 100 cây, max_depth=5, learning_rate=0,1
- **Kết quả**: R²=0,26, RMSE=12,26, MAE=9,87
- **Cải Thiện**: R² tốt hơn 13%, RMSE tốt hơn 6,1%
- **Diễn giải**: Giải thích 26% phương sai

### 10. So Sánh Mô Hình (Cell 33)
- So sánh chỉ số song song
- Trực quan hóa biểu đồ cột
- **Người Chiến Thắng**: XGBoost ở tất cả chỉ số

### 11. Độ Quan Trọng Đặc Trưng (Cell 34)
- Trích xuất độ quan trọng từ XGBoost
- 5 yếu tố dự báo hàng đầu:
  1. bữa trưa (34,2%)
  2. trình độ cha mẹ (21,5%)
  3. luyện thi (18,9%)
  4. chủng tộc/dân tộc (1,9%)
  5. giới tính (1,1%)
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
- Bảng điều khiển: Thống kê, chỉ số
- Bảng: DataFrames được hiển thị
- Biểu đồ: 5+ trực quan hóa (EDA, so sánh, độ quan trọng)
- Cảnh báo: An toàn để bỏ qua (không dùng nữa)

---

## 💾 Các Biến Chính

### Sau Tiền Xử Lý (Cell 21)
- `X`: Đặc trưng (1000 × 11)
- `y`: Biến mục tiêu (1000,)
- `X_train`, `X_test`: Chia train/test (800/200)
- `y_train`, `y_test`: Chia biến mục tiêu

### Sau Mô Hình Hóa (Cells 29, 32)
- `lr_model`: Đối tượng Hồi Quy Tuyến Tính
- `xgb_model`: Đối tượng XGBoost
- `y_pred_lr`, `y_pred_xgb`: Dự đoán
- `lr_metrics`, `xgb_metrics`: Từ điển kết quả

### Sau Phân Tích Độ Quan Trọng (Cell 34)
- `feature_importance`: DataFrame với xếp hạng

---

## 🎨 Trực Quan Hóa Được Tạo

1. **Biểu đồ cột** (Cell 16) - Phân phối điểm
2. **Biểu đồ nhiệt** (Cell 17) - Ma trận tương quan
3. **Biểu đồ hộp** (Cell 18) - Đặc trưng vs điểm toán
4. **Biểu đồ cột** (Cell 33) - So sánh mô hình
5. **Biểu đồ cột** (Cell 34) - Độ quan trọng đặc trưng

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
