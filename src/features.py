from sklearn.preprocessing import StandardScaler
import numpy as np

def transform_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    X_train['Amount_scaled'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount_scaled'] = scaler.transform(X_test[['Amount']])

    X_train['Amount_log'] = np.log1p(X_train['Amount'])
    X_test['Amount_log'] = np.log1p(X_test['Amount'])

    return X_train, X_test, scaler
