import pandas as pd
import seaborn as sns
import matplotlib as plt
from fontTools.diff import summarize

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

print(df.dtypes) #Column Types are as they should be

print(df.describe()) #describes ranges for tenure, monthly charges, and total charges which may be important for prediction.


numeric_vars = df[["tenure","MonthlyCharges","TotalCharges", "Churn"]]#Churn for hue

sns.pairplot(data = numeric_vars, hue = "Churn")
plt.pyplot.show()

categorical_columns = df.drop(columns = ["tenure","MonthlyCharges","TotalCharges"])
categorical_col_names = list(categorical_columns.columns)


for name in categorical_col_names:
   sns.countplot(data = categorical_columns, x = name, hue = "Churn")
   plt.pyplot.show()




