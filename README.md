# Student Performance Prediction - Machine Learning
> XGBoost Regression for Predicting Student Math Achievement

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Overview

A machine learning project predicting student mathematics performance using demographic, socioeconomic, and behavioral factors. The project includes exploratory data analysis, model development with XGBoost, and data-driven policy recommendations.

**Key Metrics**:
- Dataset: 395 students with 30 independent features
- Models: Linear Regression (baseline) vs XGBoost
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
├── README.md                     # Main overview
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── SOURCE/
│   ├── brave9.ipynb             # Main Jupyter notebook (48+ cells)
│   ├── README.md                # Notebook guide
│   └── student-mat.csv          # Dataset (395 students)
├── REPORT/
│   ├── mainver2.pdf             # Academic report (28 pages)
│   ├── mainver2.tex             # LaTeX source
│   ├── tailieu.bib              # Bibliography
│   └── Sections/                # Report components
└── SLIDE/                        # Presentation slides
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Bravee9/Student-Performance-Prediction-XGBoost.git
cd Student-Performance-Prediction-XGBoost

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook SOURCE/brave9.ipynb
```

---

## Dataset Information

- **Source**: Student Math Performance from Portuguese secondary schools
- **Samples**: 395 students
- **Features**: 30 independent variables
  - Demographic: school, sex, age, address, famsize, Pstatus
  - Family: Medu, Fedu, Mjob, Fjob, guardian, famrel
  - Academic: studytime, failures, schoolsup, famsup, paid, activities, nursery, higher
  - Behavioral: traveltime, absences, internet, romantic, freetime, goout, Dalc, Walc, health
  - Other: reason
- **Target**: G3 (final math grade, 0-20 scale)
- **Data Quality**: Zero missing values

---

## Models Comparison

| Metric | Linear Regression | XGBoost | Winner |
|--------|-------------------|---------|--------|
| **R² Score** | 0.230 | 0.263 | XGBoost |
| **RMSE** | 12.53 | 12.26 | XGBoost |
| **MAE** | 10.12 | 9.87 | XGBoost |

**XGBoost Configuration**:
- 100 trees with max_depth=5, learning_rate=0.1
- Subsample=0.8, Colsample_bytree=0.8
- Random state: 42, Objective: reg:squarederror

---

## Key Insights & Policy Recommendations

### 1. Past Performance is Strongest Predictor
- G2 (second period grade): 28.5%
- G1 (first period grade): 26.8%
- **Recommendation**: Implement early warning systems for students with G1 < 10

### 2. Academic Behaviors Matter
- Past failures: 12.3%
- Study time: 8.2%
- Absences: 6.1%
- **Recommendation**: Provide failure recovery and study support programs

### 3. Family Background Impact
- Parental education: ~6%
- School support: 1.8%
- **Recommendation**: Engage parents in educational activities

---

## Technical Stack

```
pandas==2.0.3           # Data manipulation
numpy==1.24.3           # Numerical computing
scikit-learn==1.3.0     # ML algorithms
xgboost==1.7.6          # Gradient boosting
matplotlib==3.7.2       # Visualization
seaborn==0.12.2         # Statistical graphics
jupyter==1.0.0          # Interactive notebooks
scipy==1.11.1           # Scientific computing
```

---

## Author & Contact

| Field | Info |
|-------|------|
| **Name** | Bui Quang Chien |
| **Student ID** | 23001837 |
| **Email** | 23001837@hus.edu.vn |
| **GitHub** | [@Bravee9](https://github.com/Bravee9) |

---

## License

MIT License - See [LICENSE](LICENSE) file

```bibtex
@misc{StudentPerfPrediction2025,
  author = {Chien, Bui Quang},
  title = {Student Performance Prediction using XGBoost},
  year = {2025},
  school = {Hanoi University of Science},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bravee9/Student-Performance-Prediction-XGBoost}}
}
```

---

## References

- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Cortez, P., & Silva, A. (2008). "Using Data Mining to Predict Secondary School Student Performance"
- [Scikit-learn Docs](https://scikit-learn.org/)
- [XGBoost Docs](https://xgboost.readthedocs.io/)

---

<div align="center">

**Last Updated**: May 2026  
**Status**: Complete

</div>
