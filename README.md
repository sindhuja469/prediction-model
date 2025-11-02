# prediction-model
A loan default prediction assignment from my coursework
# 💰 Loan Default Prediction — Machine Learning Project

## Project Overview
This project aims to predict the likelihood of a borrower defaulting on a loan using machine learning models.  
The analysis leverages the **Give Me Some Credit** dataset from Kaggle, which contains financial and demographic data of borrowers.

The primary goal is to:
- Identify individuals who are at higher risk of default.
- Help financial institutions reduce loan losses through proactive risk assessment.
- Build interpretable, data-driven models to support fair and transparent lending decisions.

---

## Objectives
1. Explore and clean the credit dataset.
2. Handle **class imbalance** (~7% defaults) using class weighting and model parameters.
3. Build predictive models:
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost (gradient boosting)
4. Evaluate performance using ROC-AUC, PR-AUC, and confusion matrices.
5. Recommend the best model for production deployment.

---

## Dataset
**Source:** [Give Me Some Credit – Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
## Challenges & Approach
The project is organized into several analytical stages:
---

### **Challenge 1 — Data Preparation & Exploration**
- Loaded and inspected the Kaggle dataset (`cs-training.csv`).
- Conducted Exploratory Data Analysis (EDA) to understand feature distributions.
- Visualized class imbalance (93% non-defaults vs. 7% defaults).

### **Challenge 2 — Model Building & Imbalance Handling**
- Created preprocessing pipelines using `ColumnTransformer`.
- Applied:
  - `class_weight="balanced"` for Logistic Regression & Random Forest.
  - `scale_pos_weight ≈ 14` for XGBoost.
- Tested all models on the same stratified train/test split.

### **Challenge 3 — Evaluation & Interpretation**
- Compared model performance using ROC-AUC and PR-AUC metrics.
- Generated classification reports and confusion matrices.
- Identified trade-offs between recall (detecting defaulters) and precision (avoiding false positives).

---

## Key Results

| Model | ROC-AUC | PR-AUC |
|--------|----------|---------|
| Logistic Regression | 0.802 | 0.327 |
| Random Forest | **0.841** | **0.354** |
| XGBoost | 0.834 | 0.342 |

**Findings:**
- Random Forest achieved the best overall performance.
- XGBoost closely followed with strong probability calibration.
- Logistic Regression provided interpretability, suitable for compliance and transparency.

---

## Technologies Used
- **Python** (pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn)
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Google Colab / Jupyter Notebook

---

## Insights
- Class imbalance strongly impacts model behavior — accuracy alone is misleading.
- Class weighting improved recall for minority (default) cases.
- Ensemble models (RF, XGBoost) captured nonlinear patterns better than linear models.
- Logistic Regression remains valuable for regulatory explainability.

---

## Ethical Considerations
- No sensitive or protected attributes were used (e.g., gender, race, ethnicity).
- Predictions should **support**, not replace, human decisions in lending.
- Transparency and fairness are prioritized through explainable model design.

---

##  How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/sindhuja469/prediction-model.git
   cd prediction-model
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # requirements.txt
   pandas==2.2.2
   numpy==1.26.4
   scikit-learn==1.5.2
   xgboost==2.1.1
   imbalanced-learn==0.12.3
   matplotlib==3.9.2
   seaborn==0.13.2
   shap==0.45.0
   joblib==1.4.2
3.Run the notebook or script:
   ```bash
   jupyter notebook loan_default.ipynb
