"""
Test the RandomForest class on the wine quality datasets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.RandomForest import RandomForest
from utils import categorical_accuracy, mae_accuracy, mse_accuracy, max_dif

ds_path_1 = '../data/winequality-red_NO_ALCOHOL.csv'
ds_path_2 = '../data/winequality-white_NO_ALCOHOL.csv'

# Load the datasets
df1 = pd.read_csv(ds_path_1, sep=';')
df2 = pd.read_csv(ds_path_2, sep=';')


def test_random_forest(dataset: pd.DataFrame):
    # Split the dataset into training and testing sets
    X = dataset.drop('quality', axis=1).values
    y = dataset['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model
    forest = RandomForest(n_trees=100, max_depth=10, min_samples_split=2)
    forest.fit(X_train, y_train)

    # Predict the target values
    y_pred = forest.predict(X_test)

    # Calculate the accuracy
    m = max_dif(y)
    print(f'Categorical accuracy: {categorical_accuracy(y_test, y_pred)}')
    print(f'MAE accuracy: {mae_accuracy(m, y_test, y_pred)}')
    print(f'MSE accuracy: {mse_accuracy(m, y_test, y_pred)}')


if __name__ == '__main__':
    print('Red wine dataset:')
    test_random_forest(df1)

    print('\n--------------------------------\n')

    print('White wine dataset:')
    test_random_forest(df2)
