# Student Performance Prediction - Machine Learning
# Dự Đoán Kết Quả Học Tập của Học Sinh - Học Máy

> **XGBoost Regression for Predicting Student Math Achievement**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Language Preference / Chọn Ngôn Ngữ

- **[ENGLISH](#english-version)** - Main documentation (scroll down for full English version)
- **[TIẾNG VIỆT](#vietnamese-version)** - Tài liệu tiếng Việt (cuộn xuống để xem phiên bản tiếng Việt đầy đủ)

---

# ENGLISH VERSION

## 🎯 Project Overview

A comprehensive machine learning project that predicts student mathematics achievement using demographic and socioeconomic factors. The project includes exploratory data analysis, model development, and policy recommendations based on data-driven insights.

**Key Metrics**:
- Dataset: 1,000 students with 8 features
- Models: Linear Regression (baseline) vs XGBoost (main)
- Results: XGBoost R² = 0.26, 13% improvement over baseline

### Key Findings
- **Top Predictor**: Lunch status/SES (34.2% importance)
- **Education Effect**: Parental education (21.5%)
- **Intervention Impact**: Test preparation (18.9%)

---

## 🏫 Course Information

| Field | Details |
|-------|---------|
| **Course** | MAT3533 - 1 K68A3 - Machine Learning |
| **University** | Hanoi University of Science |
| **Semester** | Fall 2025-2026 |
| **Author** | Bùi Quang Chiến |
| **Student ID** | 23001837 |
| **Email** | 23001837@hus.edu.vn |

---

## 📂 Project Structure

```
├── README.md                    # Main overview (this file)
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── SOURCE/
│   ├── brave9.ipynb            # Main Jupyter notebook (37 cells)
│   ├── README.md               # Notebook cell-by-cell guide
│   └── StudentsPerformance.csv  # Dataset (1,000 students)
│
├── REPORT/
│   ├── main.pdf                # Academic report (60 pages, LaTeX)
│   ├── main.tex                # LaTeX source
│   ├── tailieu.bib             # Bibliography
│   ├── hus.sty                 # HUS LaTeX style
│   └── Sections/               # Report components
│       ├── 1-Title.tex         # Title page
│       └── Images/             # Figures & charts
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Bravee9/Student-Performance-Prediction-XGBoost.git
cd Student-Performance-Prediction-XGBoost

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate            # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter notebook
jupyter notebook SOURCE/brave9.ipynb

# 5. View academic report
# Open REPORT/main.pdf in your PDF viewer
```

---

## 📊 Dataset Information

- **Source**: [Kaggle - Student Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams)
- **Samples**: 1,000 students
- **Features**: 
  - `gender` - Student gender (male/female)
  - `race/ethnicity` - Ethnic group (A, B, C, D, E)
  - `parental level of education` - Parent education level
  - `lunch` - Lunch status (standard or free/reduced)
  - `test preparation course` - Test prep completion (completed/none)
  - `reading score` - Reading score (0-100)
  - `writing score` - Writing score (0-100)
- **Target**: `math score` (0-100)
- **Data Quality**: Zero missing values (clean dataset)

---

## 🤖 Models Comparison

| Metric | Linear Regression | XGBoost | Winner |
|--------|-------------------|---------|--------|
| **R² Score** | 0.230 | 0.260 | ✓ XGBoost |
| **RMSE** | 13.05 | 12.26 | ✓ XGBoost |
| **MAE** | 10.24 | 9.87 | ✓ XGBoost |

**XGBoost Configuration**:
- 100 trees with max_depth=5
- Learning rate: 0.1
- Subsample: 0.8, Colsample_bytree: 0.8
- Random state: 42 (for reproducibility)

---

## 💡 Key Insights & Policy Implications

### Socioeconomic Status (SES) Impact
Students with standard lunch score **10.2 points higher** than those with free/reduced lunch (15% gap). This is the strongest predictor of math achievement.

**Policy Recommendation**: Expand meal subsidy programs for maximum return on investment.

### Education Gradient
Parental education shows linear relationship with math scores, with approximately **7.4 point spread** from high school to master's degree level.

**Policy Recommendation**: Establish parent engagement and education programs to strengthen family academic support.

### Intervention Effectiveness
Test preparation courses demonstrate **5.0 point improvement** in math scores, showing that targeted interventions can be effective.

**Policy Recommendation**: Universalize access to test preparation courses, especially for disadvantaged students.

---

## 📈 Project Workflow

```
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Feature Analysis → Policy Recommendations
```

**Main Sections in Notebook**:
1. Introduction & Task Description
2. Library Setup
3. Data Loading & Basic Exploration
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing & Encoding
6. Model Evaluation Function
7. Linear Regression Baseline
8. XGBoost Main Model
9. Model Comparison
10. Feature Importance Analysis
11. Conclusions & Recommendations

---

## ⚙️ Technical Stack

```
pandas==1.3.5           # Data manipulation and analysis
numpy==1.21.6           # Numerical computing
scikit-learn==1.0.2     # Machine learning algorithms
xgboost==1.5.2          # Gradient boosting framework
matplotlib==3.5.1       # Data visualization
seaborn==0.11.2         # Statistical graphics
jupyter==1.0.0          # Interactive notebooks
```

---

## 📚 Documentation

- **SOURCE/README.md** - Detailed cell-by-cell notebook guide
- **REPORT/main.pdf** - Full academic report with methodology, results, and analysis
- This README - Project overview and quick start guide

---

## 👤 Author & Contact

| Item | Information |
|------|-----------|
| **Name** | Bùi Quang Chiến |
| **Student ID** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |
| **Facebook** | [Bùi Quang Chiến](https://www.facebook.com/buiquangchienhus/) |

---

## 📄 License & Citation

**License**: MIT (see [LICENSE](LICENSE) file)

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chiến, Bùi Quang},
  title = {Student Performance Prediction using Machine Learning},
  year = {2025},
  school = {Hanoi University of Science},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bravee9/Student-Performance-Prediction-XGBoost}},
  note = {Midterm Project MAT3533-1K68A3}
}
```

---

## 🔗 References

- Bourdieu, P. (1986). "The Forms of Capital"
- Sirin, S. R. (2005). "Socioeconomic Status and Academic Achievement: A Meta-Analytic Review"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

<div align="center">

**Last Updated**: November 2025  
**Status**: ✅ Complete & Ready for Production

⭐ **If you find this project helpful, please star the repository!** ⭐

</div>

---

---

# VIETNAMESE VERSION

# Dự Đoán Kết Quả Học Tập của Học Sinh - Học Máy
# Hồi Quy XGBoost để Dự Đoán Thành Tích Toán của Học Sinh

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Tổng Quan Dự Án

Một dự án học máy toàn diện dự đoán thành tích toán học của học sinh dựa trên các yếu tố nhân khẩu học và kinh tế-xã hội. Dự án bao gồm phân tích khám phá dữ liệu, phát triển mô hình, và các khuyến nghị chính sách dựa trên các hiểu biết từ dữ liệu.

**Các Chỉ Số Chính**:
- Bộ dữ liệu: 1.000 học sinh với 8 đặc trưng
- Mô hình: Hồi Quy Tuyến Tính (cơ sở) vs XGBoost (chính)
- Kết quả: XGBoost R² = 0.26, cải thiện 13% so với cơ sở

### Những Phát Hiện Chính
- **Yếu Tố Dự Báo Hàng Đầu**: Tình trạng bữa trưa/KXH (34.2% quan trọng)
- **Ảnh Hưởng Giáo Dục**: Trình độ học vấn cha mẹ (21.5%)
- **Tác Động Can Thiệp**: Khóa luyện thi (18.9%)

---

## 🏫 Thông Tin Khóa Học

| Trường | Chi Tiết |
|-------|---------|
| **Khóa Học** | MAT3533 - 1 K68A3 - Học Máy |
| **Trường Đại Học** | Đại Học Khoa Học Tự Nhiên, Hà Nội |
| **Học Kỳ** | Thu 2025-2026 |
| **Tác Giả** | Bùi Quang Chiến |
| **Mã Sinh Viên** | 23001837 |
| **Email** | 23001837@hus.edu.vn |

---

## 📂 Cấu Trúc Dự Án

```
├── README.md                    # Tổng quan chính (file này)
├── LICENSE                      # Giấy phép MIT
├── requirements.txt             # Thư viện Python
├── .gitignore
│
├── SOURCE/
│   ├── brave9.ipynb            # Notebook Jupyter chính (37 cells)
│   ├── README.md               # Hướng dẫn chi tiết từng cell
│   └── StudentsPerformance.csv  # Bộ dữ liệu (1.000 học sinh)
│
├── REPORT/
│   ├── main.pdf                # Báo cáo học thuật (60 trang, LaTeX)
│   ├── main.tex                # Mã nguồn LaTeX
│   ├── tailieu.bib             # Tài liệu tham khảo
│   ├── hus.sty                 # Kiểu LaTeX HUS
│   └── Sections/               # Các phần của báo cáo
│       ├── 1-Title.tex         # Trang bìa
│       └── Images/             # Hình ảnh và biểu đồ
```

---

## 🚀 Bắt Đầu Nhanh (5 phút)

### Yêu Cầu Tiên Quyết
- Python 3.8 hoặc cao hơn
- Trình quản lý pip
- Jupyter Notebook

### Cài Đặt

```bash
# 1. Clone kho lưu trữ
git clone https://github.com/Bravee9/Student-Performance-Prediction-XGBoost.git
cd Student-Performance-Prediction-XGBoost

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/macOS
# hoặc
venv\Scripts\activate            # Windows PowerShell

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 4. Khởi động Jupyter notebook
jupyter notebook SOURCE/brave9.ipynb

# 5. Xem báo cáo học thuật
# Mở REPORT/main.pdf trong trình xem PDF
```

---

## 📊 Thông Tin Bộ Dữ Liệu

- **Nguồn**: [Kaggle - Student Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams)
- **Số Mẫu**: 1.000 học sinh
- **Đặc Trưng**: 
  - `gender` - Giới tính học sinh (nam/nữ)
  - `race/ethnicity` - Nhóm chủng tộc (A, B, C, D, E)
  - `parental level of education` - Trình độ học vấn cha mẹ
  - `lunch` - Tình trạng bữa trưa (bình thường hoặc miễn phí/giảm giá)
  - `test preparation course` - Hoàn thành khóa luyện thi
  - `reading score` - Điểm đọc (0-100)
  - `writing score` - Điểm viết (0-100)
- **Biến Mục Tiêu**: `math score` (0-100)
- **Chất Lượng Dữ Liệu**: Không có giá trị thiếu (bộ dữ liệu sạch)

---

## 🤖 So Sánh Mô Hình

| Chỉ Số | Hồi Quy Tuyến Tính | XGBoost | Người Chiến Thắng |
|--------|-------------------|---------|--------|
| **Điểm R²** | 0.230 | 0.260 | ✓ XGBoost |
| **RMSE** | 13.05 | 12.26 | ✓ XGBoost |
| **MAE** | 10.24 | 9.87 | ✓ XGBoost |

**Cấu Hình XGBoost**:
- 100 cây quyết định với max_depth=5
- Tốc độ học: 0.1
- Subsample: 0.8, Colsample_bytree: 0.8
- Random state: 42 (để tái lập kết quả)

---

## 💡 Những Hiểu Biết Chính & Hàm Ý Chính Sách

### Ảnh Hưởng của Tình Trạng Kinh Tế-Xã Hội (KXH)
Học sinh có bữa trưa bình thường đạt điểm cao hơn **10.2 điểm** so với những học sinh có bữa trưa miễn phí/giảm giá (chênh lệch 15%). Đây là yếu tố dự báo mạnh nhất của thành tích toán học.

**Khuyến Nghị Chính Sách**: Mở rộng chương trình hỗ trợ bữa ăn để có tỷ suất lợi tức cao nhất.

### Độ Dốc Giáo Dục
Trình độ học vấn cha mẹ cho thấy mối quan hệ tuyến tính với điểm toán, với khoảng cách khoảng **7.4 điểm** từ cấp THPT đến thạc sĩ.

**Khuyến Nghị Chính Sách**: Thành lập các chương trình tham gia cha mẹ và giáo dục để tăng cường hỗ trợ học tập từ gia đình.

### Hiệu Quả Can Thiệp
Các khóa luyện thi cho thấy **cải thiện 5.0 điểm** về điểm toán, cho thấy các can thiệp có mục tiêu có thể hiệu quả.

**Khuyến Nghị Chính Sách**: Phổ cập việc tiếp cận các khóa luyện thi, đặc biệt là cho học sinh có hoàn cảnh khó khăn.

---

## 📈 Quy Trình Dự Án

```
Tải Dữ Liệu → EDA → Tiền Xử Lý → Huấn Luyện Mô Hình → Đánh Giá → Phân Tích Đặc Trưng → Khuyến Nghị Chính Sách
```

**Các Phần Chính trong Notebook**:
1. Giới Thiệu & Mô Tả Nhiệm Vụ
2. Thiết Lập Thư Viện
3. Tải Dữ Liệu & Khám Phá Cơ Bản
4. Phân Tích Khám Phá Dữ Liệu (EDA)
5. Tiền Xử Lý Dữ Liệu & Mã Hóa
6. Hàm Đánh Giá Mô Hình
7. Cơ Sở Hồi Quy Tuyến Tính
8. Mô Hình XGBoost Chính
9. So Sánh Mô Hình
10. Phân Tích Độ Quan Trọng Đặc Trưng
11. Kết Luận & Khuyến Nghị

---

## ⚙️ Ngăn Xếp Công Nghệ

```
pandas==1.3.5           # Thao tác và phân tích dữ liệu
numpy==1.21.6           # Tính toán số học
scikit-learn==1.0.2     # Các thuật toán học máy
xgboost==1.5.2          # Framework gradient boosting
matplotlib==3.5.1       # Trực quan hóa dữ liệu
seaborn==0.11.2         # Đồ thị thống kê
jupyter==1.0.0          # Notebook tương tác
```

---

## 📚 Tài Liệu

- **SOURCE/README.md** - Hướng dẫn chi tiết từng cell của notebook
- **REPORT/main.pdf** - Báo cáo học thuật đầy đủ với phương pháp, kết quả và phân tích
- README này - Tổng quan dự án và hướng dẫn bắt đầu nhanh

---

## 👤 Tác Giả & Liên Hệ

| Mục | Thông Tin |
|------|-----------|
| **Tên** | Bùi Quang Chiến |
| **Mã Sinh Viên** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |
| **Facebook** | [Bùi Quang Chiến](https://www.facebook.com/buiquangchienhus/) |

---

## 📄 Giấy Phép & Trích Dẫn

**Giấy Phép**: MIT (xem file [LICENSE](LICENSE))

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chiến, Bùi Quang},
  title = {Dự Đoán Kết Quả Học Tập của Học Sinh Sử Dụng Học Máy},
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
- Sirin, S. R. (2005). "Socioeconomic Status and Academic Achievement: A Meta-Analytic Review"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- [Tài Liệu Scikit-learn](https://scikit-learn.org/)
- [Tài Liệu XGBoost](https://xgboost.readthedocs.io/)

---

<div align="center">

**Cập Nhật Lần Cuối**: Tháng 11 năm 2025  
**Trạng Thái**: ✅ Hoàn Thành & Sẵn Sàng Sản Xuất

⭐ **Nếu bạn thấy dự án này hữu ích, vui lòng đánh dấu sao cho kho lưu trữ!** ⭐

</div>

---

<div align="center">

**Last Updated**: November 2025  
**Status**: ✅ Complete & Ready for Production

⭐ **If you find this project helpful, please star the repository!** ⭐

</div>

---

---

# VIETNAMESE VERSION

# Dự Đoán Kết Quả Học Tập của Học Sinh - Học Máy
# Hồi Quy XGBoost để Dự Đoán Thành Tích Toán của Học Sinh

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Tổng Quan Dự Án

Một dự án học máy toàn diện dự đoán thành tích toán học của học sinh dựa trên các yếu tố nhân khẩu học và kinh tế-xã hội. Dự án bao gồm phân tích khám phá dữ liệu, phát triển mô hình, và các khuyến nghị chính sách dựa trên các hiểu biết từ dữ liệu.

**Các Chỉ Số Chính**:
- Bộ dữ liệu: 1.000 học sinh với 8 đặc trưng
- Mô hình: Hồi Quy Tuyến Tính (cơ sở) vs XGBoost (chính)
- Kết quả: XGBoost R² = 0.26, cải thiện 13% so với cơ sở

### Những Phát Hiện Chính
- **Yếu Tố Dự Báo Hàng Đầu**: Tình trạng bữa trưa/KXH (34.2% quan trọng)
- **Ảnh Hưởng Giáo Dục**: Trình độ học vấn cha mẹ (21.5%)
- **Tác Động Can Thiệp**: Khóa luyện thi (18.9%)
