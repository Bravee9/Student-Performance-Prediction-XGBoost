# Student Performance Prediction - Machine Learning
# Dự Đoán Kết Quả Học Tập của Học Sinh - Học Máy

> XGBoost Regression for Predicting Student Math Achievement

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Language Preference / Chọn Ngôn Ngữ

- [ENGLISH](#english-version) - Main documentation (scroll down for full English version)
- [TIẾNG VIỆT](#vietnamese-version) - Tài liệu tiếng Việt (cuộn xuống để xem phiên bản tiếng Việt đầy đủ)

---

# ENGLISH VERSION

## Project Overview

A comprehensive machine learning project that predicts student mathematics performance using demographic, socioeconomic, and behavioral factors. The project includes exploratory data analysis, model development with XGBoost, and actionable policy recommendations based on data-driven insights.

**Key Metrics**:
- Dataset: 395 students with 30 independent features
- Models: Linear Regression (baseline) vs XGBoost (main)
- Results: XGBoost R² = 0.263, RMSE = 12.26, MAE = 9.87

### Key Findings
- **Past Performance**: G1 and G2 grades are the strongest predictors (55% combined importance)
- **Academic Behaviors**: Failures, study time, and absences significantly impact outcomes
- **Family Background**: Parental education and family support play important roles

---

## Course Information

| Field | Details |
|-------|---------|
| **Course** | MAT3533 - 1 K68A3 - Machine Learning |
| **University** | Hanoi University of Science |
| **Semester** | Fall 2025-2026 |
| **Author** | Bui Quang Chien |
| **Student ID** | 23001837 |
| **Email** | 23001837@hus.edu.vn |

---

## Project Structure

```
├── README.md                     # Main overview (this file)
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── .gitignore
│
├── SOURCE/
│   ├── brave9.ipynb             # Main Jupyter notebook (48+ cells)
│   ├── README.md                # Notebook cell-by-cell guide
│   └── student-mat.csv          # Dataset (395 students)
│
├── REPORT/
│   ├── mainver2.pdf             # Academic report (28 pages, LaTeX)
│   ├── mainver2.tex             # LaTeX source
│   ├── tailieu.bib              # Bibliography
│   ├── hus.sty                  # HUS LaTeX style
│   └── Sections/                # Report components
│       ├── 1-Title.tex          # Title page
│       └── Images/              # Figures and charts
│
└── SLIDE/                        # Presentation slides (if needed)
```

---

## Quick Start

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
# Open REPORT/mainver2.pdf in your PDF viewer
```

---

## Dataset Information

- **Source**: Student Math Performance from Portuguese secondary schools
- **Samples**: 395 students
- **Features (30 independent variables)**: 
  - **Demographic**: school, sex, age, address, famsize, Pstatus
  - **Family Background**: Medu, Fedu, Mjob, Fjob, guardian, famrel
  - **Academic**: studytime, failures, schoolsup, famsup, paid, activities, nursery, higher
  - **Behavioral**: traveltime, absences, internet, romantic, freetime, goout, Dalc, Walc, health
  - **Other**: reason
- **Target Variable**: G3 (final math grade, 0-20 scale)
- **Additional**: G1 (first period grade), G2 (second period grade)
- **Data Quality**: Zero missing values (clean dataset)

---

## Models Comparison

| Metric | Linear Regression | XGBoost | Winner |
|--------|-------------------|---------|--------|
| **R² Score** | 0.230 | 0.263 | XGBoost |
| **RMSE** | 12.53 | 12.26 | XGBoost |
| **MAE** | 10.12 | 9.87 | XGBoost |

**XGBoost Configuration**:
- 100 decision trees with max_depth=5
- Learning rate: 0.1
- Subsample: 0.8, Colsample_bytree: 0.8
- Random state: 42 (for reproducibility)
- Objective: reg:squarederror (MSE)

---

## Key Insights and Policy Implications

### Past Performance is the Strongest Predictor
- G2 (second period grade): 28.5% importance
- G1 (first period grade): 26.8% importance
- Combined: 55.3% of total importance

**Policy Recommendation**: Implement early warning systems to monitor G1 scores. Students with G1 < 10 should receive immediate intervention.

### Academic Behaviors Matter
- Past failures: 12.3% importance
- Study time: 8.2% importance
- Absences: 6.1% importance

**Policy Recommendation**: 
- Provide failure recovery programs
- Promote structured study habits
- Reduce absenteeism through engagement initiatives

### Family Background Impact
- Mother's education (Medu): 3.2% importance
- Father's education (Fedu): 2.9% importance
- School support (schoolsup): 1.8% importance

**Policy Recommendation**: Engage parents in educational activities and provide family support programs.

---

## Project Workflow

```
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Feature Analysis → Recommendations
```

**Main Sections in Notebook**:
1. Introduction and Research Context
2. Library Setup
3. Data Description (30 features)
4. Data Loading and Inspection
5. Exploratory Data Analysis (EDA)
6. Dimensionality Reduction (PCA and t-SNE)
7. Clustering Analysis (K-Means and GMM)
8. Data Preprocessing
9. Model Training (Linear Regression and XGBoost)
10. Model Evaluation and Comparison
11. Feature Importance Analysis
12. Conclusions and Recommendations

---

## Technical Stack

```
pandas==2.0.3           # Data manipulation and analysis
numpy==1.24.3           # Numerical computing
scikit-learn==1.3.0     # Machine learning algorithms
xgboost==1.7.6          # Gradient boosting framework
matplotlib==3.7.2       # Data visualization
seaborn==0.12.2         # Statistical graphics
jupyter==1.0.0          # Interactive notebooks
scipy==1.11.1           # Scientific computing
```

---

## Documentation

- **SOURCE/README.md** - Detailed cell-by-cell notebook guide
- **REPORT/mainver2.pdf** - Full academic report with methodology, results, and analysis
- This README - Project overview and quick start guide

---

## Author and Contact

| Item | Information |
|------|-----------|
| **Name** | Bui Quang Chien |
| **Student ID** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |
| **Facebook** | [Bui Quang Chien](https://www.facebook.com/buiquangchienhus/) |

---

## License and Citation

**License**: MIT (see [LICENSE](LICENSE) file)

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chien, Bui Quang},
  title = {Student Performance Prediction using XGBoost},
  year = {2025},
  school = {Hanoi University of Science},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bravee9/Student-Performance-Prediction-XGBoost}},
  note = {Midterm Project MAT3533-1K68A3}
}
```

---

## References

- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". KDD 2016.
- Cortez, P., & Silva, A. (2008). "Using Data Mining to Predict Secondary School Student Performance".
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine".
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

<div align="center">

**Last Updated**: November 2025  
**Status**: Complete and Ready for Production

</div>

---

---

# VIETNAMESE VERSION

# Dự Đoán Kết Quả Học Tập của Học Sinh - Học Máy
# Hồi Quy XGBoost để Dự Đoán Thành Tích Toán của Học Sinh

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Tổng Quan Dự Án

Một dự án học máy toàn diện dự đoán thành tích toán học của học sinh dựa trên các yếu tố nhân khẩu học, kinh tế-xã hội và hành vi học tập. Dự án bao gồm phân tích khám phá dữ liệu, phát triển mô hình XGBoost, và các khuyến nghị chính sách dựa trên phân tích dữ liệu.

**Các Chỉ Số Chính**:
- Bộ dữ liệu: 395 học sinh với 30 đặc trưng độc lập
- Mô hình: Hồi Quy Tuyến Tính (cơ sở) vs XGBoost (chính)
- Kết quả: XGBoost R² = 0.263, RMSE = 12.26, MAE = 9.87

### Những Phát Hiện Chính
- **Kết Quả Quá Khứ**: Điểm G1 và G2 là yếu tố dự báo mạnh nhất (55% tổng độ quan trọng)
- **Hành Vi Học Tập**: Thất bại, thời gian học và vắng mặt ảnh hưởng đáng kể
- **Bối Cảnh Gia Đình**: Trình độ học vấn cha mẹ và hỗ trợ gia đình đóng vai trò quan trọng

---

## Thông Tin Khóa Học

| Trường | Chi Tiết |
|-------|---------|
| **Khóa Học** | MAT3533 - 1 K68A3 - Học Máy |
| **Trường Đại Học** | Đại Học Khoa Học Tự Nhiên, Hà Nội |
| **Học Kỳ** | Thu 2025-2026 |
| **Tác Giả** | Bùi Quang Chiến |
| **Mã Sinh Viên** | 23001837 |
| **Email** | 23001837@hus.edu.vn |

---

## Cấu Trúc Dự Án

```
├── README.md                     # Tổng quan chính (file này)
├── LICENSE                       # Giấy phép MIT
├── requirements.txt              # Thư viện Python
├── .gitignore
│
├── SOURCE/
│   ├── brave9.ipynb             # Notebook Jupyter chính (48+ cells)
│   ├── README.md                # Hướng dẫn chi tiết từng cell
│   └── student-mat.csv          # Bộ dữ liệu (395 học sinh)
│
├── REPORT/
│   ├── mainver2.pdf             # Báo cáo học thuật (28 trang, LaTeX)
│   ├── mainver2.tex             # Mã nguồn LaTeX
│   ├── tailieu.bib              # Tài liệu tham khảo
│   ├── hus.sty                  # Kiểu LaTeX HUS
│   └── Sections/                # Các phần của báo cáo
│       ├── 1-Title.tex          # Trang bìa
│       └── Images/              # Hình ảnh và biểu đồ
│
└── SLIDE/                        # Slide thuyết trình (nếu cần)
```

---

## Bắt Đầu Nhanh

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
# Mở REPORT/mainver2.pdf trong trình xem PDF
```

---

## Thông Tin Bộ Dữ Liệu

- **Nguồn**: Student Math Performance từ các trường trung học ở Bồ Đào Nha
- **Số Mẫu**: 395 học sinh
- **Đặc Trưng (30 biến độc lập)**: 
  - **Nhân khẩu học**: school, sex, age, address, famsize, Pstatus
  - **Bối cảnh gia đình**: Medu, Fedu, Mjob, Fjob, guardian, famrel
  - **Học tập**: studytime, failures, schoolsup, famsup, paid, activities, nursery, higher
  - **Hành vi**: traveltime, absences, internet, romantic, freetime, goout, Dalc, Walc, health
  - **Khác**: reason
- **Biến Mục Tiêu**: G3 (điểm toán cuối năm, thang điểm 0-20)
- **Bổ sung**: G1 (điểm kỳ 1), G2 (điểm kỳ 2)
- **Chất Lượng Dữ Liệu**: Không có giá trị thiếu (bộ dữ liệu sạch)

---

## So Sánh Mô Hình

| Chỉ Số | Hồi Quy Tuyến Tính | XGBoost | Người Chiến Thắng |
|--------|-------------------|---------|--------|
| **Điểm R²** | 0.230 | 0.263 | XGBoost |
| **RMSE** | 12.53 | 12.26 | XGBoost |
| **MAE** | 10.12 | 9.87 | XGBoost |

**Cấu Hình XGBoost**:
- 100 cây quyết định với max_depth=5
- Tốc độ học: 0.1
- Subsample: 0.8, Colsample_bytree: 0.8
- Random state: 42 (để tái lập kết quả)
- Objective: reg:squarederror (MSE)

---

## Những Hiểu Biết Chính và Hàm Ý Chính Sách

### Kết Quả Quá Khứ là Yếu Tố Dự Báo Mạnh Nhất
- G2 (điểm kỳ 2): 28.5% độ quan trọng
- G1 (điểm kỳ 1): 26.8% độ quan trọng
- Tổng cộng: 55.3% tổng độ quan trọng

**Khuyến Nghị Chính Sách**: Triển khai hệ thống cảnh báo sớm để theo dõi điểm G1. Học sinh có G1 < 10 nên nhận can thiệp ngay lập tức.

### Hành Vi Học Tập Quan Trọng
- Thất bại trong quá khứ: 12.3% độ quan trọng
- Thời gian học: 8.2% độ quan trọng
- Vắng mặt: 6.1% độ quan trọng

**Khuyến Nghị Chính Sách**: 
- Cung cấp chương trình phục hồi sau thất bại
- Khuyến khích thói quen học tập có cấu trúc
- Giảm vắng mặt thông qua các sáng kiến tham gia

### Ảnh Hưởng Bối Cảnh Gia Đình
- Trình độ học vấn mẹ (Medu): 3.2% độ quan trọng
- Trình độ học vấn bố (Fedu): 2.9% độ quan trọng
- Hỗ trợ từ trường (schoolsup): 1.8% độ quan trọng

**Khuyến Nghị Chính Sách**: Thu hút cha mẹ vào các hoạt động giáo dục và cung cấp các chương trình hỗ trợ gia đình.

---

## Quy Trình Dự Án

```
Tải Dữ Liệu → EDA → Tiền Xử Lý → Huấn Luyện Mô Hình → Đánh Giá → Phân Tích Đặc Trưng → Khuyến Nghị
```

**Các Phần Chính trong Notebook**:
1. Giới Thiệu và Bối Cảnh Nghiên Cứu
2. Thiết Lập Thư Viện
3. Mô Tả Dữ Liệu (30 đặc trưng)
4. Tải Dữ Liệu và Kiểm Tra
5. Phân Tích Khám Phá Dữ Liệu (EDA)
6. Giảm Chiều Dữ Liệu (PCA và t-SNE)
7. Phân Tích Phân Cụm (K-Means và GMM)
8. Tiền Xử Lý Dữ Liệu
9. Huấn Luyện Mô Hình (Hồi Quy Tuyến Tính và XGBoost)
10. Đánh Giá và So Sánh Mô Hình
11. Phân Tích Độ Quan Trọng Đặc Trưng
12. Kết Luận và Khuyến Nghị

---

## Ngăn Xếp Công Nghệ

```
pandas==2.0.3           # Thao tác và phân tích dữ liệu
numpy==1.24.3           # Tính toán số học
scikit-learn==1.3.0     # Các thuật toán học máy
xgboost==1.7.6          # Framework gradient boosting
matplotlib==3.7.2       # Trực quan hóa dữ liệu
seaborn==0.12.2         # Đồ thị thống kê
jupyter==1.0.0          # Notebook tương tác
scipy==1.11.1           # Tính toán khoa học
```

---

## Tài Liệu

- **SOURCE/README.md** - Hướng dẫn chi tiết từng cell của notebook
- **REPORT/mainver2.pdf** - Báo cáo học thuật đầy đủ với phương pháp, kết quả và phân tích
- README này - Tổng quan dự án và hướng dẫn bắt đầu nhanh

---

## Tác Giả và Liên Hệ

| Mục | Thông Tin |
|------|-----------|
| **Tên** | Bùi Quang Chiến |
| **Mã Sinh Viên** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |
| **Facebook** | [Bùi Quang Chiến](https://www.facebook.com/buiquangchienhus/) |

---

## Giấy Phép và Trích Dẫn

**Giấy Phép**: MIT (xem file [LICENSE](LICENSE))

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chiến, Bùi Quang},
  title = {Dự Đoán Kết Quả Học Tập của Học Sinh Sử Dụng XGBoost},
  year = {2025},
  school = {Đại Học Khoa Học Tự Nhiên, Hà Nội},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bravee9/Student-Performance-Prediction-XGBoost}},
  note = {Dự Án Giữa Kỳ MAT3533-1K68A3}
}
```

---

## Tài Liệu Tham Khảo

- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". KDD 2016.
- Cortez, P., & Silva, A. (2008). "Using Data Mining to Predict Secondary School Student Performance".
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine".
- [Tài Liệu Scikit-learn](https://scikit-learn.org/)
- [Tài Liệu XGBoost](https://xgboost.readthedocs.io/)

---

<div align="center">

**Cập Nhật Lần Cuối**: Tháng 11 năm 2025  
**Trạng Thái**: Hoàn Thành và Sẵn Sàng Sử Dụng

</div>
