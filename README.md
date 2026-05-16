# 🚢 Titanic Survival Prediction — Logistic Regression

A machine learning project that predicts passenger survival on the Titanic using Logistic Regression, built with Python and scikit-learn.

---

## 📌 Overview

This project applies a binary classification model to the classic Titanic dataset. Given passenger attributes such as age, class, sex, and embarkation port, the model predicts whether a passenger survived or not.

---

## 📁 Project Structure

```
titanic-logistic-regression/
│
├── titanic.py        # Main script: preprocessing, training, evaluation & visualization
├── titanic.csv       # Dataset
└── README.md
```

---

## ⚙️ Workflow

### 1. Data Preprocessing
- **Dropped** irrelevant columns: `PassengerId`, `Name`, `Ticket`, `Cabin`, `Fare`
- **Handled missing `Age` values** using class-based median imputation:
  - 1st class → 37
  - 2nd class → 29
  - 3rd class → 24
- **Filled missing `Embarked`** values with the mode
- **Encoded** categorical variables (`Sex`, `Embarked`) using `LabelEncoder`
- **Scaled** the `Age` feature using `StandardScaler`

### 2. Model Training
- Split data: **80% training / 20% testing** (`random_state=42`)
- Model: `LogisticRegression` from scikit-learn

### 3. Evaluation
- Metrics reported:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-Score)

### 4. Visualization
- Decision boundary plotted using **Age (Scaled)** vs **Pclass**
- A secondary model is trained on these two features for a 2D visual representation

---

## 🧰 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | Preprocessing, modeling, evaluation |

---

## ▶️ How to Run

1. Clone or download this repository
2. Place `titanic.csv` in the same directory as `titanic.py` and update the file path in the script:
   ```python
   df = pd.read_csv('titanic.csv')
   ```
3. Run the script:
   ```bash
   python titanic.py
   ```

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Gender (encoded) |
| `Age` | Age in years (imputed & scaled) |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Embarked` | Port of embarkation (encoded) |

**Target:** `Survived` (0 = No, 1 = Yes)

---

## 📈 Sample Output

```
Accuracy: 0.80

Confusion Matrix:
[[90 15]
 [21 53]]

Classification Report:
              precision    recall  f1-score   support
           0       0.81      0.86      0.83       105
           1       0.78      0.72      0.75        74
    accuracy                           0.80       179
```

> *Values above are approximate and may vary slightly.*



---

## 📚 Dataset

The dataset used is the well-known [Titanic dataset](https://www.kaggle.com/c/titanic/data) available on Kaggle.

---

## 🙋 Author

Built as part of an AI/ML course — Session 16: Logistic Regression.
