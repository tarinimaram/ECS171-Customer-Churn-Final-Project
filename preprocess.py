import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop identifier column — not a predictive feature
df.drop(columns=["customerID"], inplace=True)

# TotalCharges is stored as a string; 11 rows with tenure=0 have blank values
# (new customers with no charges yet) → fill with 0.0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

# Encode target as binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encode all remaining categorical columns
# SeniorCitizen is already 0/1; all other object columns are categorical
categorical_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set:   {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set:       {X_test.shape[0]} samples")
print(f"Churn rate — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")
