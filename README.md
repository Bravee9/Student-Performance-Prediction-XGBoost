# 📚 Student Performance Prediction - Machine Learning
# 📚 Dự Đoán Kết Quả Học Tập của Học Sinh - Học Máy

> **XGBoost Regression for Predicting Student Math Achievement**
> **Hồi Quy XGBoost để Dự Đoán Thành Tích Toán của Học Sinh**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview / Tổng Quan Dự Án

A machine learning project predicting student mathematics achievement using demographic and socioeconomic factors.

Một dự án học máy dự đoán thành tích toán học của học sinh dựa trên các yếu tố nhân khẩu học và kinh tế-xã hội.

**Dataset / Bộ dữ liệu**: 1,000 students / học sinh, 8 features / đặc trưng  
**Models / Mô hình**: Linear Regression (baseline) / (cơ sở) vs XGBoost (main) / (chính)  
**Results / Kết quả**: XGBoost R² = 0.26 / 0,26, 13% improvement over baseline / cải thiện 13% so với cơ sở

### Key Findings / Những Phát Hiện Chính
- **Top Predictor / Yếu Tố Dự Báo Hàng Đầu**: Lunch status/SES / Tình trạng bữa trưa/KXH (34.2% / 34,2% importance / quan trọng)
- **Education Effect / Ảnh Hưởng Giáo Dục**: Parental education / Trình độ học vấn cha mẹ (21.5% / 21,5%)
- **Intervention / Can Thiệp**: Test preparation / Khóa luyện thi (18.9% / 18,9%)

---

## 🏫 Course Information / Thông Tin Khóa Học

| Field / Trường | Details / Chi Tiết |
|-------|---------|
| **Course / Khóa Học** | MAT3533 - 1 K68A3 - Machine Learning / Học Máy |
| **University / Trường Đại Học** | Hanoi University of Science / Đại Học Khoa Học Tự Nhiên, Hà Nội |
| **Semester / Học Kỳ** | Fall 2025-2026 |
| **Author / Tác Giả** | Bùi Quang Chiến |
| **Student ID / Mã Sinh Viên** | 23001837 |
| **Email** | 23001837@hus.edu.vn |

---

## 📂 Project Structure / Cấu Trúc Dự Án

```
├── README.md                    # This file (Main overview) / File này (Tổng quan chính)
├── LICENSE                      # MIT License / Giấy phép MIT
├── requirements.txt             # Dependencies (v2.1 Updated) / Các thư viện phụ thuộc (v2.1 Cập nhật)
├── .gitignore
│
├── SOURCE/
│   ├── brave9.ipynb            # Main notebook (37 cells, with fixes) / Notebook chính (37 cells, có sửa chữa)
│   ├── README.md               # Notebook guide / Hướng dẫn notebook
│   └── StudentsPerformance.csv  # Dataset (1000 rows) / Bộ dữ liệu (1000 hàng)
│
├── REPORT/
│   ├── main.pdf                # Academic report (57 pages) / Báo cáo học thuật (57 trang)
│   ├── main.tex                # LaTeX source / Mã nguồn LaTeX
│   ├── tailieu.bib             # Bibliography / Tài liệu tham khảo
│   ├── hus.sty                 # HUS LaTeX style / Kiểu LaTeX HUS
│   └── Sections/               # Report sections / Các phần của báo cáo
│       ├── 1-Title.tex         # Title page / Trang bìa
│       └── Images/             # Figures & charts / Hình ảnh & biểu đồ
```

---

## 🚀 Quick Start (5 minutes) / Bắt Đầu Nhanh (5 phút)

### Prerequisites / Yêu Cầu Tiên Quyết
- Python 3.8+
- Jupyter Notebook

### Installation / Cài Đặt

```bash
# 1. Clone repository
git clone https://github.com/Bravee9/Student-Performance-Prediction-XGBoost.git

# 2. Create & activate virtual environment
# Tạo & kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or / hoặc
venv\Scripts\activate            # Windows

# 3. Install dependencies
# Cài đặt các thư viện
pip install -r requirements.txt

# 4. Run notebook
# Chạy notebook
jupyter notebook SOURCE/brave9.ipynb

# 5. View report
# Xem báo cáo
# Open REPORT/main.pdf / Mở REPORT/main.pdf
```

---

## 📊 Dataset / Bộ Dữ Liệu

- **Source / Nguồn**: [Kaggle - Student Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams)
- **Samples / Số Mẫu**: 1,000 students / học sinh
- **Features / Đặc Trưng**: gender / giới tính, race/ethnicity / chủng tộc/dân tộc, parental education / trình độ học vấn cha mẹ, lunch type / loại bữa trưa, test prep / luyện thi, reading score / điểm đọc, writing score / điểm viết
- **Target / Biến Mục Tiêu**: math score / điểm toán (0-100)
- **Quality / Chất Lượng**: No missing values / Không có giá trị thiếu

---

## 🤖 Models Compared / Mô Hình So Sánh

| Metric / Chỉ Số | Linear Regression | XGBoost | Winner / Người Chiến Thắng |
|--------|-------------------|---------|--------|
| **R² Score / Điểm R²** | 0.230 | 0.260 | XGBoost ✓ |
| **RMSE** | 13.05 | 12.26 | XGBoost ✓ |
| **MAE** | 10.24 | 9.87 | XGBoost ✓ |

**XGBoost Configuration / Cấu Hình XGBoost**:
- 100 trees / cây, max_depth=5, learning_rate=0.1
- Subsample=0.8, colsample_bytree=0.8

---

## 💡 Key Insights / Những Hiểu Biết Chính

### SES Impact / Ảnh Hưởng của KXH
Students with standard lunch score **10.2 points higher** than free/reduced lunch students (15% gap).

Học sinh có bữa trưa bình thường đạt điểm toán **cao hơn 10,2 điểm** so với học sinh có bữa trưa miễn phí/giảm giá (chênh lệch 15%).

### Education Gradient / Độ Dốc Giáo Dục
Parental education shows linear relationship with math scores (**7.4 point spread** from HS to Master's).

Trình độ học vấn cha mẹ cho thấy mối quan hệ tuyến tính với điểm toán (**chênh lệch 7,4 điểm** từ THPT đến Thạc sĩ).

### Intervention Effect / Hiệu Ứng Can Thiệp
Test preparation courses show **5.0 point improvement**, demonstrating intervention effectiveness.

Các khóa luyện thi cho thấy **cải thiện 5,0 điểm**, chứng tỏ hiệu quả của can thiệp.

### Policy Recommendations / Khuyến Nghị Chính Sách
1. **Expand meal subsidy programs** (highest ROI) / **Mở rộng chương trình hỗ trợ bữa ăn** (ROI cao nhất)
2. **Parent engagement programs** (family support) / **Chương trình tham gia cha mẹ** (hỗ trợ gia đình)
3. **Universalize test preparation** (skill building) / **Phổ cập luyện thi** (xây dựng kỹ năng)

---

## 📈 Workflow / Quy Trình Công Việc

```
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Feature Analysis
Tải Dữ Liệu → EDA → Tiền Xử Lý → Huấn Luyện Mô Hình → Đánh Giá → Phân Tích Đặc Trưng
```

**Notebook Sections / Các Phần của Notebook**:
1. Introduction & Task Description / Giới Thiệu & Mô Tả Nhiệm Vụ
2. Library Setup / Thiết Lập Thư Viện
3. Data Loading & Exploration / Tải Dữ Liệu & Khám Phá
4. Statistical Analysis (EDA) / Phân Tích Thống Kê (EDA)
5. Data Preprocessing & Encoding / Tiền Xử Lý & Mã Hóa Dữ Liệu
6. Evaluation Metrics Function / Hàm Chỉ Số Đánh Giá
7. Linear Regression Baseline / Cơ Sở Hồi Quy Tuyến Tính
8. XGBoost Main Model / Mô Hình XGBoost Chính
9. Model Comparison / So Sánh Mô Hình
10. Feature Importance Analysis / Phân Tích Độ Quan Trọng Đặc Trưng
11. Conclusions & Recommendations / Kết Luận & Khuyến Nghị

---

## ⚙️ Technical Stack / Ngăn Xếp Công Nghệ

```
pandas==1.3.5           # Data manipulation / Thao Tác Dữ Liệu
numpy==1.21.6           # Numerical computing / Tính Toán Số Học
scikit-learn==1.0.2     # ML algorithms / Các Thuật Toán ML
xgboost==1.5.2          # Gradient boosting / Gradient Boosting
matplotlib==3.5.1       # Visualization / Trực Quan Hóa
seaborn==0.11.2         # Statistical graphics / Đồ Thị Thống Kê
jupyter==1.0.0          # Notebooks
```

---

## 📚 Documentation / Tài Liệu

- **SOURCE/README.md** - Cell-by-cell notebook explanation / Giải thích chi tiết từng cell của notebook
- **REPORT/main.pdf** - Full academic report (57 pages, LaTeX) / Báo cáo học thuật đầy đủ (57 trang, LaTeX)

---

## 👤 Author & Contact / Tác Giả & Liên Hệ

| | Details / Chi Tiết |
|---------|---------|
| **Name / Tên** | Bùi Quang Chiến |
| **ID / Mã SV** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |
| **Facebook** | [Bùi Quang Chiến](https://www.facebook.com/buiquangchienhus/) |

---

## 📄 License & Citation / Giấy Phép & Trích Dẫn

**License / Giấy Phép**: MIT (see / xem [LICENSE](LICENSE))

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chiến, Bùi Quang},
  title = {Student Performance Prediction using Machine Learning / Dự Đoán Kết Quả Học Tập Học Sinh Sử Dụng Học Máy},
  year = {2025},
  school = {Hanoi University of Science / Đại Học Khoa Học Tự Nhiên, Hà Nội},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bravee9/Student-Performance-Prediction-XGBoost}},
  note = {Midterm Project MAT3533-1K68A3 / Dự Án Giữa Kỳ MAT3533-1K68A3}
}
```

---

## 🔗 References / Tài Liệu Tham Khảo

- Bourdieu, P. (1986). "The Forms of Capital"
- Sirin, S. R. (2005). "Socioeconomic Status and Academic Achievement"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- [Scikit-learn](https://scikit-learn.org/) | [XGBoost](https://xgboost.readthedocs.io/)

---

<div align="center">

**Last Updated / Cập Nhật Lần Cuối**: November 2025 / Tháng 11 năm 2025  
**Status / Trạng Thái**: ✅ Complete & Ready / Hoàn Thành & Sẵn Sàng

⭐ **If helpful, please star the repository! / Nếu thấy hữu ích, vui lòng đánh dấu sao cho repository!** ⭐

</div>
