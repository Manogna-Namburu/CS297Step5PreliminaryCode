import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "data/ClaMP_Integrated-5184.csv"
SEED = 42


def prepare_data():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Remove duplicates
    original_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    print("Original samples:", original_count)
    print("Duplicates removed:", original_count - len(df))
    print("Remaining samples:", len(df))


    X = df.drop(columns=["class", "packer_type"])
    y = df["class"].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1.")

    print("Features:", X.shape[1])
    print("Class distribution:", y.value_counts().to_dict())

    # 20% test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    # Split training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,
        random_state=SEED,
        stratify=y_train
    )

    # Handle missing values
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scale features
    scaler = MinMaxScaler(clip=True)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\nTraining samples:", len(y_train))
    print("Validation samples:", len(y_val))
    print("Testing samples:", len(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    prepare_data()