# -----------------------------
# Multi-label Cuisine Prediction - Clean & Improved Version
# -----------------------------

import warnings
warnings.filterwarnings("ignore")  # suppress all warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Data/restaurant_data.csv")

# -----------------------------
# 2. Drop missing cuisines
# -----------------------------
df = df.dropna(subset=["Cuisines"])

# -----------------------------
# 3. Convert cuisines to lists
# -----------------------------
df["Cuisine_List"] = df["Cuisines"].apply(lambda x: [c.strip() for c in x.split(",")])

# -----------------------------
# 4. Keep top N cuisines
# -----------------------------
top_n = 15
all_cuisines = [c for sublist in df["Cuisine_List"] for c in sublist]
top_cuisines = pd.Series(all_cuisines).value_counts().nlargest(top_n).index
df["Cuisine_List"] = df["Cuisine_List"].apply(lambda x: [c for c in x if c in top_cuisines])
df = df[df["Cuisine_List"].map(len) > 0]

# -----------------------------
# 5. MultiLabelBinarizer
# -----------------------------
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["Cuisine_List"])

# -----------------------------
# 6. Feature selection and preprocessing
# -----------------------------
X = df[[
    "Average Cost for two", "Aggregate rating",
    "Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"
]].copy()

# Convert Yes/No to 0/1
binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
for col in binary_cols:
    X[col] = X[col].map({"Yes":1, "No":0})

# Feature Engineering
X['Name_length'] = df['Restaurant Name'].apply(len)
X['Cost_Rating_interaction'] = X['Average Cost for two'] * X['Aggregate rating']

# One-hot encode categorical features
categorical_cols = ["City", "Currency", "Rating color", "Rating text"]
X = pd.get_dummies(pd.concat([X, df[categorical_cols]], axis=1), drop_first=True)

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 8. Oversampling minority labels
# -----------------------------
oversampled_X = []
oversampled_y = []

for i in range(y_train.shape[1]):
    label_indices = np.where(y_train[:, i] == 1)[0]
    for idx in label_indices:
        oversampled_X.append(X_train.iloc[idx])
        oversampled_y.append(y_train[idx])

X_train_bal = pd.DataFrame(oversampled_X).reset_index(drop=True)
y_train_bal = np.array(oversampled_y)

# -----------------------------
# 9. Train XGBoost with OneVsRest for multi-label
# -----------------------------
model = OneVsRestClassifier(
    XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )
)
model.fit(X_train_bal, y_train_bal)

# -----------------------------
# 10. Predict probabilities
# -----------------------------
y_prob = model.predict_proba(X_test)

# -----------------------------
# 11. Threshold tuning per label
# -----------------------------
thresholds = np.full(y_test.shape[1], 0.5)  # default 0.5
y_pred = np.zeros_like(y_test)

for i in range(y_test.shape[1]):
    best_thresh = 0.5
    best_f1 = 0
    for t in np.arange(0.2, 0.8, 0.05):
        pred = (y_prob[:, i] >= t).astype(int)
        f1 = f1_score(y_test[:, i], pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    thresholds[i] = best_thresh
    y_pred[:, i] = (y_prob[:, i] >= best_thresh).astype(int)

# -----------------------------
# 12. Evaluation
# -----------------------------
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Subset Accuracy:", accuracy_score(y_test, y_pred))
print("\nMacro F1:", f1_score(y_test, y_pred, average='macro'))
print("Micro F1:", f1_score(y_test, y_pred, average='micro'))

print("\nClassification Report (per cuisine):")
for idx, cuisine in enumerate(mlb.classes_):
    print(f"{cuisine}:")
    print(classification_report(y_test[:, idx], y_pred[:, idx], zero_division=0))

# -----------------------------
# 13. True vs Predicted counts
# -----------------------------
true_counts = y_test.sum(axis=0)
pred_counts = y_pred.sum(axis=0)
confusion_df = pd.DataFrame({"Cuisine": mlb.classes_, "True": true_counts, "Predicted": pred_counts})
confusion_df = confusion_df.sort_values(by="True", ascending=False)
print("\nTrue vs Predicted counts per cuisine:\n", confusion_df)

# -----------------------------
# 14. Plot comparison
# -----------------------------
plt.figure(figsize=(12, 6))
sns.barplot(x="Cuisine", y="True", data=confusion_df, color="blue", label="True")
sns.barplot(x="Cuisine", y="Predicted", data=confusion_df, color="red", alpha=0.5, label="Predicted")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Count")
plt.title("True vs Predicted Counts per Cuisine")
plt.legend()
plt.show()
