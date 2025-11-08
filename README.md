# 📚 Dự Đoán Kết Quả Học Tập của Học Sinh - Học Máy

> **Hồi Quy XGBoost để Dự Đoán Thành Tích Toán của Học Sinh**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Tổng Quan Dự Án

Một dự án học máy dự đoán thành tích toán học của học sinh dựa trên các yếu tố nhân khẩu học và kinh tế-xã hội.

**Bộ dữ liệu**: 1.000 học sinh, 8 đặc trưng  
**Mô hình**: Hồi Quy Tuyến Tính (cơ sở) vs XGBoost (chính)  
**Kết quả**: XGBoost R² = 0,26, cải thiện 13% so với cơ sở

### Những Phát Hiện Chính
- **Yếu Tố Dự Báo Hàng Đầu**: Tình trạng bữa trưa/KXH (34,2% quan trọng)
- **Ảnh Hưởng Giáo Dục**: Trình độ học vấn cha mẹ (21,5%)
- **Can Thiệp**: Khóa luyện thi (18,9%)

---

## 🏫 Thông Tin Khóa Học

| Trường | Chi Tiết |
|-------|---------|
| **Khóa Học** | MAT3533 - 1 K68A3 - Học Máy |
| **Trường Đại Học** | Đại Học Khoa Học Tự Nhiên, Hà Nội |
| **Học Kỳ** | Fall 2025-2026 |
| **Tác Giả** | Bùi Quang Chiến |
| **Mã Sinh Viên** | 23001837 |
| **Email** | 23001837@hus.edu.vn |

---

## 📂 Cấu Trúc Dự Án

```
├── README.md                    # File này (Tổng quan chính)
├── LICENSE                      # Giấy phép MIT
├── requirements.txt             # Các thư viện phụ thuộc (v2.1 Cập nhật)
├── .gitignore
│
├── SOURCE/
│   ├── brave9.ipynb            # Notebook chính (37 cells, có sửa chữa)
│   ├── README.md               # Hướng dẫn notebook
│   └── StudentsPerformance.csv  # Bộ dữ liệu (1000 hàng)
│
├── REPORT/
│   ├── main.pdf                # Báo cáo học thuật (57 trang)
│   ├── main.tex                # Mã nguồn LaTeX
│   ├── tailieu.bib             # Tài liệu tham khảo
│   ├── hus.sty                 # Kiểu LaTeX HUS
│   └── Sections/               # Các phần của báo cáo
│       ├── 1-Title.tex         # Trang bìa
│       └── Images/             # Hình ảnh & biểu đồ

**📌 MỚI trong v2.1:** 4 file tài liệu được thêm vào để đảm bảo chất lượng

---

## 🚀 Bắt Đầu Nhanh (5 phút)

### Yêu Cầu Tiên Quyết
- Python 3.8+
- Jupyter Notebook

### Cài Đặt

```bash
# 1. Clone repository
git clone https://github.com/Bravee9/Student-Performance-Prediction-XGBoost.git

# 2. Tạo & kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/macOS
# hoặc
venv\Scripts\activate            # Windows

# 3. Cài đặt các thư viện
pip install -r requirements.txt

# 4. Chạy notebook
jupyter notebook SOURCE/brave9.ipynb

# 5. Xem báo cáo
# Mở REPORT/main.pdf
```

---

## 📊 Bộ Dữ Liệu

- **Nguồn**: [Kaggle - Student Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams)
- **Số Mẫu**: 1.000 học sinh
- **Đặc Trưng**: giới tính, chủng tộc/dân tộc, trình độ học vấn cha mẹ, loại bữa trưa, luyện thi, điểm đọc, điểm viết
- **Biến Mục Tiêu**: điểm toán (0-100)
- **Chất Lượng**: Không có giá trị thiếu

---

## 🤖 Mô Hình So Sánh

| Chỉ Số | Hồi Quy Tuyến Tính | XGBoost | Người Chiến Thắng |
|--------|-------------------|---------|--------|
| **Điểm R²** | 0,230 | 0,260 | XGBoost ✓ |
| **RMSE** | 13,05 | 12,26 | XGBoost ✓ |
| **MAE** | 10,24 | 9,87 | XGBoost ✓ |

**Cấu Hình XGBoost**:
- 100 cây, max_depth=5, learning_rate=0,1
- Subsample=0,8, colsample_bytree=0,8

---

## 💡 Những Hiểu Biết Chính

### Ảnh Hưởng của KXH
Học sinh có bữa trưa bình thường đạt điểm toán **cao hơn 10,2 điểm** so với học sinh có bữa trưa miễn phí/giảm giá (chênh lệch 15%).

### Độ Dốc Giáo Dục  
Trình độ học vấn cha mẹ cho thấy mối quan hệ tuyến tính với điểm toán (**chênh lệch 7,4 điểm** từ THPT đến Thạc sĩ).

### Hiệu Ứng Can Thiệp
Các khóa luyện thi cho thấy **cải thiện 5,0 điểm**, chứng tỏ hiệu quả của can thiệp.

### Khuyến Nghị Chính Sách
1. **Mở rộng chương trình hỗ trợ bữa ăn** (ROI cao nhất)
2. **Chương trình tham gia cha mẹ** (hỗ trợ gia đình)
3. **Phổ cập luyện thi** (xây dựng kỹ năng)

---

## 📈 Quy Trình Công Việc

```
Tải Dữ Liệu → EDA → Tiền Xử Lý → Huấn Luyện Mô Hình → Đánh Giá → Phân Tích Đặc Trưng
```

**Các Phần của Notebook**:
1. Giới Thiệu & Mô Tả Nhiệm Vụ
2. Thiết Lập Thư Viện
3. Tải Dữ Liệu & Khám Phá
4. Phân Tích Thống Kê (EDA)
5. Tiền Xử Lý & Mã Hóa Dữ Liệu
6. Hàm Chỉ Số Đánh Giá
7. Cơ Sở Hồi Quy Tuyến Tính
8. Mô Hình XGBoost Chính
9. So Sánh Mô Hình
10. Phân Tích Độ Quan Trọng Đặc Trưng
11. Kết Luận & Khuyến Nghị

---

## ⚙️ Ngăn Xếp Công Nghệ

```
pandas==1.3.5           # Thao Tác Dữ Liệu
numpy==1.21.6           # Tính Toán Số Học
scikit-learn==1.0.2     # Các Thuật Toán ML
xgboost==1.5.2          # Gradient Boosting
matplotlib==3.5.1       # Trực Quan Hóa
seaborn==0.11.2         # Đồ Thị Thống Kê
jupyter==1.0.0          # Notebooks
```

---

## 📚 Tài Liệu

- **SOURCE/README.md** - Giải thích chi tiết từng cell của notebook
- **REPORT/main.pdf** - Báo cáo học thuật đầy đủ (57 trang, LaTeX)

---

## 👤 Tác Giả & Liên Hệ

| | Chi Tiết |
|---------|---------|
| **Tên** | Bùi Quang Chiến |
| **Mã Sinh Viên** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |
| **Facebook** | [Bùi Quang Chiến](https://www.facebook.com/buiquangchienhus/) |

---

## 📄 Giấy Phép & Trích Dẫn

**Giấy Phép**: MIT (xem [LICENSE](LICENSE))

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chiến, Bùi Quang},
  title = {Dự Đoán Kết Quả Học Tập Học Sinh Sử Dụng Học Máy},
  year = {2025},
  school = {Đại Học Khoa Học Tự Nhiên, Hà Nội},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bravee9/Student-Performance-Prediction-XGBoost}},
  note = {Dự Án Giữa Kỳ MAT3533-1K68A3}
}
```

---

## 🔗 Tài Liệu Tham Khảo

- Bourdieu, P. (1986). "The Forms of Capital"
- Sirin, S. R. (2005). "Socioeconomic Status and Academic Achievement"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- [Scikit-learn](https://scikit-learn.org/) | [XGBoost](https://xgboost.readthedocs.io/)

---

<div align="center">

**Cập Nhật Lần Cuối**: Tháng 11 năm 2025  
**Trạng Thái**: ✅ Hoàn Thành & Sẵn Sàng

⭐ **Nếu thấy hữu ích, vui lòng đánh dấu sao cho repository!** ⭐

</div>
