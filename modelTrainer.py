import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Keep only relevant columns
df = df[["pclass", "age", "sex", "survived"]]

# Drop missing values
df = df.dropna()

# Encode 'sex' column (male=0, female=1)
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# Features & target
X = df[["pclass", "age", "sex"]]
y = df["survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "titanic_model.pkl")
print("Model saved as titanic_model.pkl")
