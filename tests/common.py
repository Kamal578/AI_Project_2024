from sklearn.model_selection import train_test_split

red_path = '../data/winequality-red_NO_ALCOHOL.csv'
white_path = '../data/winequality-white_NO_ALCOHOL.csv'


def get_data(df, split=0.2):
    """
    Split it into training and testing sets

    :param df: the dataframe to split
    :param split: the proportion of the data to use for training
    :return: X_train, X_test, y_train, y_test
    """
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    return train_test_split(X, y, test_size=split)
