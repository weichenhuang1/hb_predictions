from load import load
from train_and_eval import train_with_best_hyperparams, eval
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

hot, sushi, drinks = load("data.csv")

hot_model = train_with_best_hyperparams(hot[0], hot[2])
sushi_model = train_with_best_hyperparams(sushi[0], sushi[2])
drinks_model = train_with_best_hyperparams(drinks[0], drinks[2])

joblib.dump(hot_model, "best_xgb_hot.pkl")
joblib.dump(sushi_model, "best_xgb_sushi.pkl")
joblib.dump(drinks_model, "best_xgb_drinks.pkl")

eval(hot_model, hot[1], hot[3])
eval(sushi_model, sushi[1], sushi[3])
eval(drinks_model, drinks[1], drinks[3])

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