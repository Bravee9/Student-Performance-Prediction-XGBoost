# 📚 Student Performance Prediction - Machine Learning

> **XGBoost Regression for Predicting Student Math Achievement**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview

A machine learning project predicting student mathematics achievement using demographic and socioeconomic factors. 

**Dataset**: 1,000 students, 8 features  
**Models**: Linear Regression (baseline) vs XGBoost (main)  
**Results**: XGBoost R² = 0.26, 13% improvement over baseline

### Key Findings
- **Top Predictor**: Lunch status/SES (34.2% importance)
- **Education Effect**: Parental education (21.5%)
- **Intervention**: Test preparation (18.9%)

---

## 🏫 Course Information

| Field | Details |
|-------|---------|
| **Course** | MAT3533 - 1 K68A3 - Học Máy |
| **University** | Hanoi University of Science |
| **Semester** | Fall 2025-2026 |
| **Author** | Bùi Quang Chiến |
| **Student ID** | 23001837 |
| **Email** | 23001837@hus.edu.vn |

---

## 📂 Project Structure

```
├── README.md                    # This file (Main overview)
├── LICENSE                      # MIT License
├── requirements.txt             # Dependencies (v2.1 Updated)
├── SETUP_GUIDE.md              # Installation & execution guide (NEW)
├── AUDIT_REPORT.md             # Quality assurance report (NEW)
├── CHANGELOG.md                # Version history & fixes (NEW)
├── FIXES_SUMMARY.md            # Summary of all fixes (NEW)
├── .gitignore
│
├── SOURCE/
│   ├── brave9.ipynb            # Main notebook (36 cells, with fixes)
│   ├── README.md               # Notebook guide
│   └── StudentsPerformance.csv  # Dataset (1000 rows)
│
├── REPORT/
│   ├── main.pdf                # Academic report (40+ pages)
│   ├── main.tex                # LaTeX source
│   ├── tailieu.bib             # Bibliography
│   ├── hus.sty                 # HUS LaTeX style
│   └── Sections/               # Report sections
│       ├── 1-Title.tex         # Title page
│       └── Images/             # Figures & charts
│
└── [DOCS/]                     # Optional documentation
```

**📌 NEW in v2.1:** 4 documentation files added for better quality assurance

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Bravee9/student-performance-prediction.git

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate            # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebook
jupyter notebook SOURCE/brave9.ipynb

# 5. View report
# Open REPORT/main.pdf
```

---

## 📊 Dataset

- **Source**: [Kaggle - Student Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams)
- **Samples**: 1,000 students
- **Features**: gender, race/ethnicity, parental education, lunch type, test prep, reading score, writing score
- **Target**: math score (0-100)
- **Quality**: No missing values

---

## 🤖 Models Compared

| Metric | Linear Regression | XGBoost | Winner |
|--------|-------------------|---------|--------|
| **R² Score** | 0.230 | 0.260 | XGBoost ✓ |
| **RMSE** | 13.05 | 12.26 | XGBoost ✓ |
| **MAE** | 10.24 | 9.87 | XGBoost ✓ |

**XGBoost Configuration**:
- 100 trees, max_depth=5, learning_rate=0.1
- Subsample=0.8, colsample_bytree=0.8

---

## 💡 Key Insights

### SES Impact
Students with standard lunch score **10.2 points higher** than free/reduced lunch students (15% gap).

### Education Gradient  
Parental education shows linear relationship with math scores (**7.4 point spread** from HS to Master's).

### Intervention Effect
Test preparation courses show **5.0 point improvement**, demonstrating intervention effectiveness.

### Policy Recommendations
1. **Expand meal subsidy programs** (highest ROI)
2. **Parent engagement programs** (family support)
3. **Universalize test preparation** (skill building)

---

## 📈 Workflow

```
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Feature Analysis
```

**Notebook Sections**:
1. Introduction & Task Description
2. Library Setup
3. Data Loading & Exploration
4. Statistical Analysis (EDA)
5. Data Preprocessing & Encoding
6. Evaluation Metrics Function
7. Linear Regression Baseline
8. XGBoost Main Model
9. Model Comparison
10. Feature Importance Analysis
11. Conclusions & Recommendations

---

## ⚙️ Technical Stack

```
pandas==1.3.5           # Data manipulation
numpy==1.21.6           # Numerical computing
scikit-learn==1.0.2     # ML algorithms
xgboost==1.5.2          # Gradient boosting
matplotlib==3.5.1       # Visualization
seaborn==0.11.2         # Statistical graphics
jupyter==1.0.0          # Notebooks
```

---

## 📚 Documentation

- **SOURCE/README.md** - Cell-by-cell notebook explanation
- **DOCS/SETUP.md** - Detailed installation & troubleshooting
- **DOCS/EDA_SUMMARY.md** - Statistical findings & analysis
- **REPORT/main.pdf** - Full academic report (40 pages, LaTeX)

---

## 👤 Author & Contact

| | Details |
|---------|---------|
| **Name** | Bùi Quang Chiến |
| **ID** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |
| **Facebook** | [Bùi Quang Chiến](https://www.facebook.com/buiquangchienhus/) |

---

## 📄 License & Citation

**License**: MIT (see [LICENSE](LICENSE))

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chiến, Bùi Quang},
  title = {Student Performance Prediction using Machine Learning},
  year = {2025},
  school = {Hanoi University of Science},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bravee9/student-performance-prediction}},
  note = {Midterm Project MAT3533-1K68A3}
}
```

---

## 🔗 References

- Bourdieu, P. (1986). "The Forms of Capital"
- Sirin, S. R. (2005). "Socioeconomic Status and Academic Achievement"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- [Scikit-learn](https://scikit-learn.org/) | [XGBoost](https://xgboost.readthedocs.io/)

---

<div align="center">

**Last Updated**: November 2025  
**Status**: ✅ Complete & Ready

⭐ **If helpful, please star the repository!** ⭐

</div>
