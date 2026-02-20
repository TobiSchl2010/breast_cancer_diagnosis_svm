from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(test_size=0.2, random_state=42):
    bunch_data = datasets.load_breast_cancer()
    X, y = (
        bunch_data.data,
        1 - bunch_data.target,
    )  # Invert target labels: 0 for benign, 1 for malignant

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
