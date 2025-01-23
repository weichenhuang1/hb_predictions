# Weather-Aware Restaurant Sales Forecasting with XGBoost

This project forecasts daily restaurant sales (for Hot Food, Sushi, and Drinks) using historical weather data and calendar information. By accurately predicting sales, managers can optimize inventory, reduce waste, and improve overall operational efficiency.

---

## Table of Contents
1. [Overview](#overview)  
2. [Data](#data)  
3. [Approach](#approach)  
4. [Hyperparameter Tuning](#hyperparameter-tuning)  
5. [Usage](#usage)  
6. [Results](#results)   
7. [License](#license)

---

## Overview
- **Motivation**: Restaurant sales can fluctuate due to weather conditions (temperature, precipitation, humidity) and time-based factors (day of week, holidays). Predictive modeling can help align food supply with demand, minimizing waste and stockouts.  
- **Goal**: Build a robust, automated forecasting pipeline to predict daily sales for different categories (Hot Food, Sushi, Drinks) using **XGBoost**.

---

## Data
- **Source**:  
  - **Sales Data**: Extracted from the Point-of-Sale (POS) system.  
  - **Weather Data**: Gathered from https://www.weatherapi.com/history/q/dallas-2654932
- **Columns** :
  1. **Date**: e.g., `YYYY-MM-DD`
  2. **Temperature**: Daily average temperature
  3. **Precipitation**: Total rainfall (in inches or mm)
  4. **Humidity**: Average daily humidity (%)
  5. **Day_of_Week**: Categorical or numeric code for Monday, Tuesday, etc.
  6. **Holiday**: Flag (0 or 1) indicating a holiday
  7. **Hot Food Sales**: Target (continuous) for hot food
  8. **Sushi Sales**: Target (continuous) for sushi
  9. **Drinks Sales**: Target (continuous) for drinks
  10. **Last_3_Days_Hot**: Rolling average of the last 3 days of hot food sales
  11. **Last_3_Days_Sushi**: Rolling average of the last 3 days of sushi sales
  12. **Last_3_Days_Drinks**: Rolling average of the last 3 days of drink sales

## Approach
1. **Data Preparation**  
   - Cleaned and preprocessed the CSV data.  
   - Converted categorical variables (`Day_of_Week`) to numerical codes and ensured `Holiday` is binary.  
   - Ensured that features and targets are aligned (no missing or misaligned dates).
   - Created a running average of the sales for the last 3 days for each food category

2. **Feature Engineering**  
   - Kept core weather features: `Temperature`, `Precipitation`, `Humidity`.  
   - Included time-based features: `Day_of_Week`, `Holiday`, 
   - Added rolling averages for sales from the last 3 days: `Last_3_Days_Hot`, `Last_3_Days_Sushi`, `Last_3_Days_Drinks`

3. **Modeling**  
   - Used **XGBoost Regressor** with `objective='reg:squarederror'`.  
   - Set up a parameter grid varying `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` etc.  
   - Performed **grid search** with **cross-validation** to find optimal hyperparameters.

4. **Evaluation**  
   - Metrics: **RMSE**, **MAE**, **R²** on the **test set**.  

5. **Deployment**  
   - The best model for each food type is saved to a `.pkl` file (via `joblib.dump`).  
   - You can load and run predictions on new data using the same pipeline in `just_predict.py`.

## Hyperparameter Tuning
- **GridSearchCV**:  
  - Explored various combinations of `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, etc.  
  - Used `scoring='neg_mean_squared_error'` 
  - 3-fold cross-validation used to balance training time and reliability.

- **Best Model**:  
  - The grid search returns a best estimator, which is then evaluated on the hold-out test set.  

## Results
In house, we saw a ~25% decrease in waste over my last 3 months employed there.

## License
This project is offered under the [MIT License](LICENSE). Feel free to modify and distribute as needed.
