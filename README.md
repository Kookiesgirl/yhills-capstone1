# yhills-capstone1
MYSELF VIVISHA CATHERIN, I HAVE DEVELOPED THE GIVEN CAPSTONE PROJECT-1 

CUSTOMER CHURN PREDICTION


📌 Customer Churn Prediction – Code Explanation
my code is a Customer Churn Prediction Model that follows key Machine Learning steps: Data Preprocessing, Model Training, Evaluation, and Interpretability.

1️⃣ Importing Necessary Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import shap
What this does:
✔ Loads essential libraries for data manipulation (Pandas, NumPy)
✔ Imports ML models (Logistic Regression, Random Forest, XGBoost)
✔ Imports metrics for evaluation (Accuracy, Precision, Recall, ROC-AUC)
✔ Uses SHAP (SHapley Additive Explanations) to explain feature importance

2️⃣ Loading the Dataset
python
Copy
Edit
df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.head())  # Display first 5 rows
✔ Loads the dataset into a Pandas DataFrame
✔ Prints first few rows to understand the data

3️⃣ Dropping Unnecessary Columns
python
Copy
Edit
df.drop(columns=["customerID"], inplace=True, errors="ignore")
✔ Removes "customerID" column (not useful for prediction)

4️⃣ Encoding Categorical Variables
python
Copy
Edit
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if df[col].nunique() == 2:  # Binary columns
        df[col] = LabelEncoder().fit_transform(df[col])
    else:  # One-hot encoding for multi-category columns
        df = pd.get_dummies(df, columns=[col], drop_first=True)
✔ Label Encoding for binary categories (e.g., "Yes"/"No" → 0/1)
✔ One-Hot Encoding for multi-category variables (e.g., "Contract Type")

5️⃣ Handling Missing Values
python
Copy
Edit
df.fillna(df.median(), inplace=True)
✔ Fills missing values with median (avoids bias in numerical features)

6️⃣ Splitting Data into Features (X) & Target (y)
python
Copy
Edit
X = df.drop(columns=["Churn"])  # Features
y = df["Churn"]  # Target variable
✔ X contains all features except "Churn"
✔ y contains the target variable ("Churn")

7️⃣ Train-Test Split (80-20)
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
✔ Splits data into 80% training, 20% testing
✔ Stratify ensures class balance (same % of "Yes"/"No" in train & test sets)

8️⃣ Feature Scaling (Standardization)
python
Copy
Edit
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
✔ Standardizes numerical features (makes them have mean=0, std=1)
✔ Helps models like Logistic Regression & XGBoost perform better

9️⃣ Model Training & Evaluation
python
Copy
Edit
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Make predictions

    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
✔ Trains three models:

Logistic Regression (Baseline model, interpretable)
Random Forest (Handles non-linearity well)
XGBoost (Boosted trees for best performance)
✔ Evaluates models using:

Accuracy → % of correct predictions
Precision → % of predicted "Yes" that were correct
Recall → % of actual "Yes" correctly predicted
ROC AUC → Measures classifier performance
Confusion Matrix → Shows actual vs predicted counts
🔟 Feature Importance with SHAP (XGBoost)
python
Copy
Edit
explainer = shap.Explainer(models["XGBoost"])
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)
✔ SHAP (SHapley Values) shows how each feature impacts churn predictions
✔ Summary Plot highlights most important features affecting churn

🎯 Final Summary
✔ Loads & Preprocesses Data (Encoding, Handling Missing Data, Scaling)
✔ Splits Data & Trains Models (Logistic Regression, Random Forest, XGBoost)
✔ Evaluates Performance (Accuracy, Precision, Recall, ROC AUC)
✔ Uses SHAP to Interpret Feature Importance

🚀 Next Steps to Improve
✅ Check for Class Imbalance (Use SMOTE if needed)
✅ Hyperparameter Tuning (GridSearchCV for optimal settings)
✅ Feature Selection (Drop less relevant features for better efficiency)
✅ Deploy Model (Use Flask or Streamlit for a web app)

