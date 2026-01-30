# Netflix Customer Churn Prediction 🔥

An end-to-end Machine Learning project to predict whether a Netflix user will churn or not based on user activity, subscription details, and engagement patterns.

This project covers the full ML workflow:
EDA → preprocessing → Logistic Regression baseline → Random Forest tuning → final evaluation + threshold tuning.

---

## 📌 Problem Statement
Customer churn is a major business challenge for subscription-based platforms.

**Goal:** Build a classification model that predicts:
- `1` → Customer will churn  
- `0` → Customer will not churn

---

## 📂 Dataset Overview
**Dataset:** `netflix_customer_churn.csv`  
✅ No missing values (0 null values)

### Columns
- `customer_id` *(dropped - identifier only)*
- `age`
- `gender`
- `subscription_type`
- `watch_hours`
- `last_login_days`
- `region`
- `device`
- `monthly_fee`
- `churned` *(Target)*
- `payment_method`
- `number_of_profiles`
- `avg_watch_time_per_day`
- `favorite_genre`

### Target distribution
Churn classes are almost balanced:
- `churned = 1` → 2515
- `churned = 0` → 2485

---

## 🔎 Exploratory Data Analysis (EDA)
Key steps performed:
- Checked null values, duplicates, datatypes
- Distribution plots for numeric features
- Count plots for categorical features
- Correlation heatmap (numeric)
- Pairplot to observe feature relationships
- Outlier handling for skewed features using:
  - `log1p()` transformation
  - winsorization / capping (percentile-based)

---

## ⚙️ Feature Engineering & Preprocessing
- Dropped `customer_id`
- Log transformed skewed variables:
  - `watch_hours` → `log_watch_hours`
  - `avg_watch_time_per_day` → `log_avg_watch_time_per_day`
- One Hot Encoding for categorical features using `OneHotEncoder(handle_unknown="ignore")`
- Used `ColumnTransformer` + `Pipeline` for leakage-safe preprocessing

---

## 🤖 Models Trained

### 1) Logistic Regression (Baseline)
**Results (Test):**
- Accuracy: **0.894**
- ROC-AUC: **0.972**

This model was used as a baseline and for interpretability (feature coefficients).

---

### 2) Random Forest (Final Model ✅)
Hyperparameter tuned using **RandomizedSearchCV** with scoring:
- `roc_auc` (threshold independent and best for churn)

**Results (Test):**
- Accuracy: **0.98**
- ROC-AUC: **0.998**

Confusion Matrix:
[[489 8]
[ 10 493]]


✅ Extremely strong performance with very low false positives and false negatives.

---

## 📈 Model Evaluation
Evaluation metrics used:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix
- ROC Curve (AUC)
- Precision-Recall Curve (AP)
- Threshold tuning for business-based cutoff selection

---

## 🧠 Key Insights
Top churn indicators observed:
- `last_login_days` ↑ → churn probability increases strongly
- Engagement metrics reduce churn:
  - higher `log_watch_hours` ↓ churn
  - higher `log_avg_watch_time_per_day` ↓ churn
- Certain payment methods and plans show churn association (category-level behavior)

---

## ✅ Final Outcome
📌 **Final model selected:** Tuned Random Forest  
📌 **Saved model pipeline:** `final_tuned_rf_churn_model.pkl`

This pipeline can directly take raw input features and output churn predictions (with chosen threshold).

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

