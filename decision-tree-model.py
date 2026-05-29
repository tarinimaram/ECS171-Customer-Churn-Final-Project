import preprocess as pp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


X_train, X_test = pp.X_train, pp.X_test
y_train, y_test = pp.y_train, pp.y_test

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=20,
    random_state=42,
)

# trains the model using the training data from preprocess.py
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# evaluation metrics
# achieved abt 80% accuracy on test set
print("Decision Tree Model")
print("=" * 40)
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# show which variables the decision tree used most for when splitting customers into churn/no churn grps
feature_importance = sorted(
    zip(X_train.columns, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
)

print("\nTop 10 Most Important Features:")
for feature, importance in feature_importance[:10]:
    print(f"{feature}: {importance:.4f}")
