import os
import numpy as np
import pandas as pd
## Function to load NASA SMAP and MSL datasets
def load_nasa_msl_data(data_label: str) -> tuple[np.array, np.array, np.array]:
    """
    Load train and test data from local.
    """

    X_train = np.load(os.path.join("data", "train", "{}.npy").format(data_label))
    X_test = np.load(os.path.join("data", "test", "{}.npy").format(data_label))
    y_test = pd.read_csv("data/labeled_anomalies.csv")

    X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], -1))
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], -1))
    y_test = y_test.loc[y_test['chan_id'] == data_label].reset_index(drop=True)


    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)


    return X_train, X_test, y_test