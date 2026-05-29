import math

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


sns.pairplot(data = numeric_vars, hue = "Churn", diag_kind = "hist")
plt.pyplot.show()

categorical_columns = df.drop(columns = ["tenure","MonthlyCharges","TotalCharges", "customerID"])
categorical_col_names = list(categorical_columns.columns)


for name in categorical_col_names:
   sns.countplot(data = categorical_columns, x = name, hue = "Churn")
   plt.pyplot.title(f"{name} Count by Churn")
   plt.pyplot.show()



#Because we have multiple categorical variables, in which we are interested in seeing how they effect another cat var. In this
#Case we can use Cramers V score, which essentially assumes and tests independence of Churn to the other dependent variables.
#Which then allows us to estimate association.


from scipy.stats import chi2_contingency


cramer_scores = {}
for name in categorical_col_names:
    if name == "Churn":
        continue

    cont_table = pd.crosstab(df[name], df["Churn"])
    n = len(df[name])
    row, col = cont_table.shape
    k = min(row, col)

    chi_square_val, p, degree_of_freedom, exp = chi2_contingency(cont_table)
    cramer_val = math.sqrt(chi_square_val/(n * (k-1)))
    cramer_scores[name] = cramer_val



cramer_val_frame = pd.DataFrame(list(cramer_scores.items()), columns = ["Categorical Variables", "Cramer Value"]).sort_values("Cramer Value", ascending = False)

sns.barplot(cramer_val_frame, y = "Categorical Variables", x = "Cramer Value")
plt.pyplot.show()

#From this plot we can see that contract, online security, tech support, internet service, and payment method have the highest levels of association with churn, and may be good to look into later.


numeric_varnames = ["MonthlyCharges", "tenure", "TotalCharges"]

for name in numeric_varnames:
    sns.boxplot(data = df, x = name, y = "Churn")
    plt.pyplot.title(f"{name} by Churn")
    plt.pyplot.show()

sns.heatmap(df[numeric_varnames].corr(), annot = True)
plt.pyplot.title("Numeric Correlation Plot")
plt.pyplot.show()

#While these correlations seem to make sense in relation to one another, from the boxplot we see interesting interactions with some numeric variables, especially tenure length, and monthly charges.


#calculating Churn rate for each category

df["ChurnBinary"] = df["Churn"].map({"Yes": 1, "No": 0})

for name in categorical_col_names:
    if name == "Churn":
        continue
    Rate = df.groupby(name, as_index = False)["ChurnBinary"].mean()
    sns.barplot(data = Rate, x = name, y = "ChurnBinary")
    plt.pyplot.title(f"{name} By Churn Rate")
    plt.pyplot.show()



#Churn rate for tenure:
churnRateByTenure = df.groupby("tenure", as_index= False)["ChurnBinary"].mean()

sns.lineplot(data = churnRateByTenure, x = "tenure", y = "ChurnBinary")
plt.pyplot.title("Churn Rate by Tenure Length")
plt.pyplot.show()

'''
Some Takeaways:

Contract has big association with Churn, followed by a few other categorical variables. (Check cramers score section for top results)

Large proportion of customer leave early in their tenure.

Automatic payment types have much lower rates of customer churn.

Smaller but still notable takeaways still exist in many of the rate graphs, most impactfull part of the EDA was the cramers results showing the most associated variables. 
'''
