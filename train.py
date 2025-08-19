import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load Titanic dataset
df = sns.load_dataset("titanic")
df = df[["pclass", "age", "sex", "survived"]]
df = df.dropna()

# Encode 'sex' column
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# Features & target
X = df[["pclass", "age", "sex"]].values
y = df["survived"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Build simple neural network
model = Sequential([
    Dense(16, activation="relu", input_shape=(3,)),  # input: pclass, age, sex
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")  # output: survival probability
]) 
# Input → 3 numbers: pclass, age, sex
# First layer → 16 neurons calculate 16 different combinations of inputs
# Hidden layer → 8 neurons mix those 16 numbers into 8 new signals
# Output layer → 1 neuron outputs a probability (0–1)


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)
# the model goes over the training set 50 times.
# batch_size=16 → updates weights after 16 samples.
# validation_split=0.2 → 20% of training data used to monitor validation performance.
# verbose=1 → shows training progress.

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Save model
model.save("titanic_nn.h5")
print("Model saved as titanic_nn.h5")
