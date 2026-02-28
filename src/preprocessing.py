# src/preprocessing.py

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_data(df, test_split=0.2, random_seed=42):

    X = df.drop("label", axis=1)
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_split,
        random_state=random_seed,
        stratify=y
    )

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    return X_train, X_test, y_train_cat, y_test_cat
