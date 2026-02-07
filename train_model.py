import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("car_data.csv")   # <-- your dataset name

# ===============================
# 2. Features & Target
# ===============================
X = df.drop("price", axis=1)
y = df["price"]

# ===============================
# 3. One-Hot Encoding
# ===============================
X = pd.get_dummies(X, drop_first=True)

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. Scaling Numerical Columns
# ===============================
numeric_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ===============================
# 6. MODEL 1 – Ridge Regression
# ===============================
model1 = Ridge(alpha=1.0)
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
print("Model 1 (Ridge) R2 Score:", r2_score(y_test, y_pred1))

# ===============================
# 7. MODEL 2 – Lasso Regression
# ===============================
model2 = Lasso(alpha=0.01)
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
print("Model 2 (Lasso) R2 Score:", r2_score(y_test, y_pred2))

# ===============================
# 8. SAVE MODELS & SCALER
# ===============================
pickle.dump(model1, open("model1.pkl", "wb"))
pickle.dump(model2, open("model2.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ model1.pkl, model2.pkl and scaler.pkl created successfully!")

# Store scores
models = ['Ridge Regression', 'Lasso Regression']
scores = [r2_score(y_test, y_pred1), r2_score(y_test, y_pred2)]

# Plot
plt.figure()
plt.bar(models, scores)
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
plt.show()