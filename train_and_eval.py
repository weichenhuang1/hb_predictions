import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def train_with_best_hyperparams(X, y):
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=21)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # or 'neg_root_mean_squared_error', 'r2', etc.
        cv=3,         
        n_jobs=3,     
        verbose=1
    )
    grid_search.fit(X, y)

    # 7. Retrieve the best model
    best_model = grid_search.best_estimator_

    return best_model


def eval(model, X_test, y_test, name=""):
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"--- {name} Model ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.3f}")
    print("")