# Task6_House_Price_Prediction.ipynb

# ------------------------------
# 1. Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ------------------------------
# 2. Load Dataset
# ------------------------------
df = pd.read_csv("house_prices.csv")
df.fillna(0, inplace=True)  # Replace missing values with 0

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
display(df.head())

# ------------------------------
# 3. Feature Selection
# ------------------------------
# Select relevant features for prediction
features = ['median_income', 'housing_median_age', 'total_bedrooms', 
            'households', 'total_rooms', 'population', 'latitude', 'longitude']
target = 'median_house_value'

X = df[features]
y = df[target]

# ------------------------------
# 4. Split Dataset
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 5. Train Model
# ------------------------------
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# 6. Model Evaluation
# ------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("--- Model Evaluation ---")
print("MAE  :", mae)
print("RMSE :", rmse)

# ------------------------------
# 7. Feature Importance
# ------------------------------
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Top Features ---")
display(importance)

# ------------------------------
# 8. Plot Actual vs Predicted Prices
# ------------------------------
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.savefig("task6_plot.png")
plt.show()

# ------------------------------
# 9. Save Model
# ------------------------------
joblib.dump(model, "house_price_model.pkl"
