import time

import pandas as pd

from src.DecisionTree import DecisionTreeRegressor
from src.RandomForest import RandomForestRegressor

from utils.metrics import mae, mse, rmse

from common import *

red_df = pd.read_csv(red_path, sep=';')
white_df = pd.read_csv(white_path, sep=';')


def test_regressor(regressor, df):
    X_train, X_test, y_train, y_test = get_data(df)

    start = time.time()
    regressor.fit(X_train, y_train)
    print('Training time: ', time.time() - start)

    # Predict the target values
    y_pred = regressor.predict(X_test)

    return mae(y_test, y_pred), mse(y_test, y_pred), rmse(y_test, y_pred)


def main():
    regressors = [
        DecisionTreeRegressor(max_depth=10),
        RandomForestRegressor(n_estimators=10, max_depth=10, name='Random Forest: 10 trees'),
        RandomForestRegressor(n_estimators=10, max_depth=10, max_features=5, name='Random Forest: 10 trees, 5 features'),
        RandomForestRegressor(n_estimators=100, max_depth=10, name='Random Forest: 100 trees'),
    ]

    dataframes = {
        'Red Wine Quality Dataset': red_df,
        'White Wine Quality Dataset': white_df
    }

    for name, df in dataframes.items():
        print('Examining ', name)
        print('---------------------------------')

        for reg in regressors:
            print('Regressor:', str(reg))

            mae_score, mse_score, rmse_score = test_regressor(reg, df)

            print('Mean Absolute Error:', mae_score)
            print('Mean Squared Error:', mse_score)
            print('Root Mean Squared Error:', rmse_score)
            print()


if __name__ == '__main__':
    main()
