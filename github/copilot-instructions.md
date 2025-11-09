# GitHub Copilot Instructions for Student Performance Prediction Project

**Project**: Student Performance Prediction using XGBoost Regression  
**Author**: Bùi Quang Chiến (23001837)  
**Course**: MAT3533 - 1 K68A3 - Machine Learning  
**University**: Hanoi University of Science

---

## 🎯 Project Focus

This project focuses on **4 main files** that must be kept synchronized and consistent:

### 1. **Notebook**: `SOURCE/brave9.ipynb` (Jupyter Notebook)
- **Purpose**: Main machine learning pipeline with 37 cells
- **Language**: 100% English (all comments, output, documentation)
- **Structure**: EDA → Preprocessing → Modeling → Evaluation → Analysis
- **Key Libraries**: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
- **Output**: Cell-by-cell ML workflow with visualizations and metrics

### 2. **Report**: `REPORT/main.tex` (LaTeX Academic Report)
- **Purpose**: Comprehensive 57-page academic report on ML methodology
- **Language**: 100% Vietnamese (all sections in Vietnamese)
- **Structure**: Introduction → Theory → Experiments → Conclusions → Appendices
- **Key Sections**:
  - Part 1: Mở đầu và Cơ sở lý thuyết (Introduction & Theory)
  - Part 2: Thực nghiệm và Phân tích (Experiments & Analysis)
  - Part 3: Kết luận và Tài liệu (Conclusions & References)
  - Appendix A: **Code và Thuật toán** (must sync with notebook)
  - Appendix B: Bảng số liệu chi tiết (Detailed tables)

### 3. **Root Documentation**: `README.md` (Root Level)
- **Purpose**: Project overview and quick start guide
- **Language**: Bilingual (separate ENGLISH VERSION + VIETNAMESE VERSION)
- **Structure**: 
  - Language preference section
  - Complete ENGLISH VERSION (all sections)
  - Complete VIETNAMESE VERSION (mirror of English)
- **Content**: Course info, project structure, installation, dataset, models, insights, policies

### 4. **Notebook Guide**: `SOURCE/README.md` (Notebook Documentation)
- **Purpose**: Cell-by-cell guide for `brave9.ipynb`
- **Language**: Bilingual (separate ENGLISH VERSION + VIETNAMESE VERSION)
- **Structure**:
  - Language preference section
  - Complete ENGLISH VERSION (12 notebook sections detailed)
  - Complete VIETNAMESE VERSION (mirror of English)
- **Content**: Each section documented with purpose, code details, variables, outputs

---

## 🔄 Synchronization Requirements

### Rule 1: Appendix A in LaTeX ↔ Notebook Code Sync

**File**: `REPORT/main.tex` (Lines 1248-1345 approx)  
**Section**: `\chapter{Phụ lục A: Code và Thuật toán}`

The code samples in Appendix A must exactly match the actual Python code from the notebook:

#### Subsections to Keep Synchronized:

1. **One-Hot Encoding** (Phụ lục A: Mã hóa Biến Phân loại)
   - Notebook Cell: Data Preprocessing section
   - Must match: `pd.get_dummies()` implementation
   - Status: ✅ Currently uses generic example, should use exact notebook code

2. **Train-Test Split** (Phụ lục A: Chia dữ liệu)
   - Notebook Cell: Lines 450-471 (Data split section)
   - Must match: `train_test_split()` parameters and usage
   - Current in LaTeX: Generic template
   - **Action Needed**: Replace with actual notebook code:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(
         X_encoded, y, test_size=0.2, random_state=42, shuffle=True
     )
     ```

3. **Linear Regression** (Phụ lục A: Linear Regression)
   - Notebook Cell: Lines 534-547
   - Must match: Model initialization, training, evaluation
   - Current in LaTeX: Generic template
   - **Action Needed**: Replace with actual notebook code

4. **XGBoost** (Phụ lục A: XGBoost)
   - Notebook Cell: Lines 556-603
   - Must match: All hyperparameters and explanations
   - Current in LaTeX: Generic template
   - **Action Needed**: Replace with actual notebook code including all parameters:
     ```python
     xgb_model = XGBRegressor(
         objective='reg:squarederror',
         n_estimators=100,
         max_depth=5,
         learning_rate=0.1,
         subsample=0.8,
         colsample_bytree=0.8,
         random_state=42,
         n_jobs=-1,
         verbosity=0
     )
     ```

5. **Feature Importance** (Phụ lục A: Phân tích Feature Importance)
   - Notebook Cell: Feature importance extraction and visualization
   - Must match: How importance is calculated and plotted
   - Current in LaTeX: Generic matplotlib example
   - **Action Needed**: Review and sync with notebook implementation

#### Synchronization Checklist:
- [ ] All hyperparameters match between notebook and LaTeX
- [ ] All variable names are consistent
- [ ] All comments/docstrings are in English (notebook) and can be translated
- [ ] Output format matches between notebook and LaTeX examples
- [ ] No contradictions in parameter explanations

---

### Rule 2: Notebook Structure ↔ README.md Sync

**Files**: 
- Notebook: `SOURCE/brave9.ipynb`
- Guide: `SOURCE/README.md`

The README must accurately reflect notebook structure:

#### Sections (12 Main Parts):
1. **Introduction** ✅ Aligned
2. **Libraries** ✅ Aligned
3. **Task Description** ✅ Aligned
4. **Data Loading** ✅ Aligned
5. **Exploratory Data Analysis** ✅ Aligned
6. **Data Preprocessing** ✅ Aligned
7. **Evaluation Function** ✅ Aligned
8. **Linear Regression** ✅ Aligned
9. **XGBoost Regression** ✅ Aligned
10. **Model Comparison** ✅ Aligned
11. **Feature Importance** ✅ Aligned
12. **Conclusions** ✅ Aligned

#### Synchronization Checklist:
- [ ] All 12 sections documented
- [ ] Cell numbers/ranges accurate
- [ ] Code explanations match notebook docstrings
- [ ] Output types match actual notebook outputs
- [ ] Variable names and shapes are correct

---

### Rule 3: README Bilingual Consistency

**Files**:
- `README.md` (root)
- `SOURCE/README.md`

Both README files must follow same bilingual structure:

#### Format Requirements:
```markdown
# Title (English & Tiếng Việt)

## 📖 Language Preference / Chọn Ngôn Ngữ
- **[ENGLISH](#english-version)** - Main documentation
- **[TIẾNG VIỆT](#vietnamese-version)** - Tài liệu tiếng Việt

---

# ENGLISH VERSION
[Complete English section with all content]

---

# VIETNAMESE VERSION
[Complete Vietnamese section with all content]
```

#### Bilingual Consistency Checklist:
- [ ] Separate ENGLISH VERSION and VIETNAMESE VERSION sections
- [ ] NO mixed "English / Tiếng Việt" on same lines
- [ ] All content in English version is complete
- [ ] All content in Vietnamese version mirrors English exactly
- [ ] Both versions are grammatically correct
- [ ] Code examples are identical in both versions
- [ ] Tables are duplicated (not shared)

---

## 📋 Content Guidelines

### Notebook Code Style (brave9.ipynb)
- Language: **100% English**
- Comments: English only
- Print statements: English only
- Variable names: English (e.g., `math_score`, `feature_importance`)
- Docstrings: English and descriptive
- Output text: English only

### LaTeX Report Style (main.tex)
- Language: **100% Vietnamese**
- All section headings: Vietnamese
- All explanations: Vietnamese
- Code samples: Python code (universal), but docstrings/comments in English match notebook
- All narrative: Vietnamese only

### README Style (Both files)
- Language: **Bilingual with clean separation**
- Structure: ENGLISH VERSION section, then VIETNAMESE VERSION section
- No mixing of languages on same line
- Both versions complete and self-contained

---

## 🔍 Quality Assurance Checks

### Before Committing Any Changes:

#### 1. Code Synchronization
- [ ] Any code change in notebook is reflected in LaTeX Appendix A?
- [ ] Hyperparameters in notebook match LaTeX examples?
- [ ] Function signatures and variable names consistent?

#### 2. Documentation Consistency
- [ ] README sections match notebook structure?
- [ ] All 12 notebook parts documented in README?
- [ ] Variable names and output types accurate?

#### 3. Language Purity
- [ ] Notebook: 100% English code/comments?
- [ ] LaTeX: 100% Vietnamese narrative?
- [ ] README: Clean bilingual separation (no mixing)?

#### 4. Bilingual Quality
- [ ] Both versions complete and self-contained?
- [ ] Vietnamese translation is accurate and idiomatic?
- [ ] Code examples identical in both versions?
- [ ] Tables/lists consistent across languages?

---

## 📝 Version Control Best Practices

### Commit Message Format
```
v{VERSION}: {Concise description of changes}

Example:
v2.5: Restructure README with separate bilingual sections
v2.6: Synchronize Appendix A code with notebook implementation
```

### Commit Checklist
- [ ] All 4 files reviewed if change affects them
- [ ] No conflicts between files
- [ ] Language standards maintained
- [ ] Synchronization rules followed
- [ ] Clear, descriptive commit message

---

## 🚀 Future Development Priorities

1. **Code Appendix Sync** (HIGH PRIORITY)
   - Replace generic code samples in Appendix A
   - Ensure exact match with notebook implementation
   - Add detailed hyperparameter explanations

2. **Enhanced README Documentation** (MEDIUM PRIORITY)
   - Add more detailed examples to SOURCE/README.md
   - Include output examples for each section
   - Document expected variable types and shapes

3. **Cross-Reference Improvements** (MEDIUM PRIORITY)
   - Add section cross-references between files
   - Link notebook sections to report sections
   - Create reference table for all 4 files

4. **LaTeX to Notebook Alignment** (LOW PRIORITY)
   - Ensure report diagrams match notebook visualizations
   - Verify all reported metrics match notebook outputs
   - Check table data for consistency

---

## 📞 Contact & Maintenance

- **Author**: Bùi Quang Chiến
- **Student ID**: 23001837
- **Email**: 23001837@hus.edu.vn
- **Repository**: https://github.com/Bravee9/Student-Performance-Prediction-XGBoost

---

**Last Updated**: November 2025  
**Status**: 🔄 Under Active Maintenance  
**Focus**: 4-File Synchronization & Quality Assurance
