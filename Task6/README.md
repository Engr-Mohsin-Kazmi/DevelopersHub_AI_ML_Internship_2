# Task 6: House Price Prediction

## Objective
Predict house prices using property features such as size, number of bedrooms, location, and other numeric attributes. The goal is to train a regression model to estimate the median house value.

## Dataset
- Source: [Housing Dataset - HandsOnML](https://github.com/ageron/handson-ml2)
- File: `house_prices.csv`
- Features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity

## Model
- Algorithm: Linear Regression
- Library: scikit-learn
- Train/Test Split: 80% train, 20% test
- Features: Numeric columns only

## Results
- MAE  : 52,615
- RMSE : 72,443
- Top Features:
  - median_income: 40,538  
  - housing_median_age: 1,183  
  - total_bedrooms: 116  

## Outputs
- `task6_plot.png` → Actual vs Predicted scatter plot  
- `house_price_model.pkl` → Saved trained model  

## How to Run
```bash
# Activate virtual environment
source venv/bin/activate

# Run the backend script
python3 task6_backend.py
