# 🏥 Predicting Hospital Readmission in Diabetic Patients

**Author:** chayma Akkachi  
**Course:** Machine Learning  Project  
**Date:** 15/04/2026

---

## 📌 Problem Statement

Unplanned hospital readmissions represent one of the most critical challenges in modern healthcare. Among all patient populations, **diabetic patients face a disproportionately high readmission risk** due to the complexity of their condition and the need for continuous post-discharge follow-up.

**Business Objective:** Provide healthcare institutions with a data-driven tool to identify, at the time of discharge, which diabetic patients are at high risk of being readmitted within 30 days.

**Technical Objective:** Build a supervised binary classification model that predicts whether a diabetic patient will be readmitted within 30 days of discharge.

---

## 📂 Dataset

| Property | Value |
|---|---|
| **Name** | Diabetes 130-US Hospitals (1999–2008) |
| **Source** | UCI Machine Learning Repository |
| **Link** | https://archive.ics.uci.edu/dataset/296 |
| **Rows** | 101,766 patient records |
| **Columns** | 50 features |
| **Target** | `readmitted` → converted to binary (< 30 days = 1, else = 0) |
| **Missing values** | Yes — coded as `?` |
| **Size** | ~20 MB |

---

## 🗂️ Repository Structure

```
├── data/
│   └── download_data.py          # Script to download dataset from UCI
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   └── 02_Modeling.ipynb         # Preprocessing, Modeling & Evaluation
├── src/
│   ├── preprocessing.py          # Reusable preprocessing functions
│   └── evaluation.py             # Reusable evaluation functions
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── README.md                     # This file
└── .gitignore                    # Files to ignore
```

---

## ⚙️ Installation

### Option 1 — pip
```bash
git clone https://github.com/yourusername/diabetes-readmission
cd diabetes-readmission
pip install -r requirements.txt
```

### Option 2 — Conda
```bash
conda env create -f environment.yml
conda activate diabetes-ml
```

### Run the notebooks
```bash
jupyter notebook notebooks/
```

> **Note:** Download `diabetic_data.csv` from the UCI link above and place it in the `data/` folder before running.

---

## 🔬 Why Classification (not Regression)?

The target variable `readmitted` is **categorical** (Yes/No within 30 days), making this a **binary classification** problem. Regression would be inappropriate here because:
- There is no continuous output to predict (no numerical value like a price or temperature)
- The clinical decision is binary: the patient **will** or **will not** be readmitted
- We need probabilities and a decision threshold, not a continuous score

---

## 🤖 Models Used & Justification

| Model | Type | Why chosen |
|---|---|---|
| **Logistic Regression** | Linear | Simple baseline; interpretable coefficients; fast to train |
| **Decision Tree** | Tree-based | Interpretable rules; handles mixed data types well |
| **Random Forest** | Ensemble (Bagging) | Reduces overfitting vs single tree; robust to noise |
| **Gradient Boosting** | Ensemble (Boosting) | High performance on tabular data; handles imbalance well |

---

## 📊 Results Summary

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |

> Results will be populated after running `02_Modeling.ipynb`

---

## 📈 Key Findings

- **`number_inpatient`** is the strongest predictor of readmission
- **`time_in_hospital`** and **`num_medications`** are strong secondary predictors
- **`insulin`** and **`diabetesMed`** carry significant clinical signal
- Class imbalance (89% vs 11%) addressed using **SMOTE** on the training set only
- Best model selected by **ROC-AUC** on the validation set, evaluated on a held-out test set

---

## 📋 Evaluation Metrics

Given the severe class imbalance (only 11% positive class), **accuracy alone is misleading**. The project uses:
- **F1-Score** — balances precision and recall
- **ROC-AUC** — measures discriminative ability across all thresholds
- **Confusion Matrix** — visualizes true/false positives and negatives
- **Precision & Recall** — critical in a medical context (missing a high-risk patient = costly)
