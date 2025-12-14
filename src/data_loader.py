import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/creditcard.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    # Time se Hour nikalna
    df['Hour'] = (df['Time'] // 3600) % 24

    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
