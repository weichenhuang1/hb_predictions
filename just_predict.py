import pandas as pd
import joblib

hot_model = joblib.load("best_xgb_hot.pkl")
sushi_model = joblib.load("best_xgb_sushi.pkl")
drinks_model = joblib.load("best_xgb_drinks.pkl")

# 2. Prepare new data for inference (must match feature format)
next_week = pd.read_csv("future.csv")

hot_forecast = hot_model.predict(next_week)
sushi_forecast = sushi_model.predict(next_week)
drinks_forecast = drinks_model.predict(next_week)

predictions = pd.DataFrame({
    'Day_of_Week': next_week['Day_of_Week'],
    'Hot Food Forecast': hot_forecast,
    'Sushi Forecast': sushi_forecast,
    'Drinks Forecast': drinks_forecast
})

predictions.to_csv("predictions.csv", index=False)