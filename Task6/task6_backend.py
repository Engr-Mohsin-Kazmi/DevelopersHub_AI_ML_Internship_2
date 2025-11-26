import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("house_prices.csv").fillna(0)

# Use numeric columns only
X = df.drop("median_house_value", axis=1)
X = X.select_dtypes(include=[np.number])
y = df["median_house_value"]

# Train/Test split
Xt, Xs, yt, ys = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression().fit(Xt, yt)
pred = model.predict(Xs)

# Metrics
print("\n--- Model Evaluation ---")
print("MAE  :", mean_absolute_error(ys, pred))
print("RMSE :", np.sqrt(mean_squared_error(ys, pred)))

# Scatter plot
sns.scatterplot(x=ys, y=pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted House Prices")
plt.savefig("task6_plot.png")
print("\nPlot saved as task6_plot.png ✅")

# Feature importance
feat_imp = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\n--- Top Features ---")
print(feat_imp.head(10))

# Save model
joblib.dump(model, "house_price_model.pkl")
print("\nModel Saved as house_price_model.pkl ✅")


