import pandas as pd
from sklearn.model_selection import train_test_split

def load(filepath):
    df = pd.read_csv(filepath)
    df['Day_of_Week'] = df['Day_of_Week'].astype('category').cat.codes

    X = df[['Temperature', 'Precipitation', 'Humidity', 'Day_of_Week', 'Holiday', "Last_3_Days"]]
    y_hot_food = df['Hot Food Sales']
    y_sushi = df['Sushi Sales']
    y_drinks = df['Drinks Sales']
    
    X_train_hot, X_test_hot, y_train_hot, y_test_hot = train_test_split(
        X, y_hot_food, test_size=0.2, random_state=21
    )

    X_train_sushi, X_test_sushi, y_train_sushi, y_test_sushi = train_test_split(
        X, y_sushi, test_size=0.2, random_state=21
    )

    X_train_drinks, X_test_drinks, y_train_drinks, y_test_drinks = train_test_split(
        X, y_drinks, test_size=0.2, random_state=21
    )

    return [[X_train_hot, X_test_hot, y_train_hot, y_test_hot], [X_train_sushi, X_test_sushi, y_train_sushi, y_test_sushi], [X_train_drinks, X_test_drinks, y_train_drinks, y_test_drinks]]