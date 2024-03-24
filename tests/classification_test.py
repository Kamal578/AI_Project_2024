import time

import pandas as pd

from src.DecisionTree import DecisionTreeClassifier
from src.RandomForest import RandomForestClassifier

from utils.metrics import accuracy, precision, recall, f1_score

from common import *

red_df = pd.read_csv(red_path, sep=';')
white_df = pd.read_csv(white_path, sep=';')


def test_classifier(classifier, df):
    X_train, X_test, y_train, y_test = get_data(df)

    start = time.time()
    classifier.fit(X_train, y_train)
    print('Training time: ', time.time() - start)

    # Predict the target values
    y_pred = classifier.predict(X_test)

    return accuracy(y_test, y_pred), precision(y_test, y_pred), recall(y_test, y_pred), f1_score(y_test, y_pred)


def main():
    classifiers = [
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(n_estimators=10, max_depth=10, name='Random Forest: 10 trees'),
        RandomForestClassifier(n_estimators=10, max_depth=10, max_features=5, name='Random Forest: 10 trees, 5 features'),
        RandomForestClassifier(n_estimators=100, max_depth=10, name='Random Forest: 100 trees'),
    ]

    dataframes = {
        'Red Wine Quality Dataset': red_df,
        'White Wine Quality Dataset': white_df
    }

    for name, df in dataframes.items():
        print('Examining ', name)
        print('---------------------------------')

        for clf in classifiers:
            print('Classifier:', str(clf))

            acc, prec, rec, f1 = test_classifier(clf, df)

            print('Accuracy:', acc)
            print('Precision:', prec)
            print('Recall:', rec)
            print('F1-score:', f1)
            print()


if __name__ == '__main__':
    main()
