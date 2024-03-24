"""
Test the DecisionTree class on the wine quality datasets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.DecisionTree import DecisionTree

ds_path_1 = '../data/winequality-red_NO_ALCOHOL.csv'
ds_path_2 = '../data/winequality-white_NO_ALCOHOL.csv'

# Load the datasets
df1 = pd.read_csv(ds_path_1, sep=';')
df2 = pd.read_csv(ds_path_2, sep=';')


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def max_dif(y):
    return np.max(y) - np.min(y)


def test_decision_tree(dataset: pd.DataFrame):
    # Split the dataset into training and testing sets
    X = dataset.drop('quality', axis=1).values
    y = dataset['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model
    tree = DecisionTree(max_depth=10, min_samples_split=2)
    tree.fit(X_train, y_train)

    # Predict the target values
    y_pred = tree.predict(X_test)

    # Calculate the accuracy
    m = max_dif(y)
    categorical_accuracy = np.mean(y_pred == y_test)
    mae_accuracy = 1 - mae(y_test, y_pred) / m
    mse_accuracy = 1 - mse(y_test, y_pred) / m ** 2

    print(f'Categorical accuracy: {categorical_accuracy}')
    print(f'MAE accuracy: {mae_accuracy}')
    print(f'MSE accuracy: {mse_accuracy}')


if __name__ == '__main__':
    print('Red wine dataset:')
    test_decision_tree(df1)

    print('\n--------------------------------\n')

    print('White wine dataset:')
    test_decision_tree(df2)
