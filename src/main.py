import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def perform_eda(df: pd.DataFrame) -> None:
    print(df.head())
    print(df.info())
    print(df.describe())

    corr_matrix = df.corr(numeric_only=True)
    price_corr = corr_matrix["Price"].sort_values(ascending=False)

    print("\nCorrelation with Price:")
    print(price_corr)

def prepare_data(df):
    df = df.drop("Address", axis=1)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    return X, y

def train_linear_regression(X_train, X_test, y_train, y_test):
    lr = LinearRegression()

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    return lr

def print_metrics(y_true, y_pred):
    print(f"MAE: {mean_absolute_error(y_true, y_pred):,.0f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):,.0f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")

def predict_house_price(model):
    income = float(input("Enter Avg. Area Income: "))
    house_age = float(input("Enter Avg. Area House Age: "))
    rooms = float(input("Enter Avg. Area Number of Rooms: "))
    bedrooms = float(input("Enter Avg. Area Number of Bedrooms: "))
    population = float(input("Enter Area Population: "))

    house_data = pd.DataFrame({
    "Avg. Area Income": [income],
    "Avg. Area House Age": [house_age],
    "Avg. Area Number of Rooms": [rooms],
    "Avg. Area Number of Bedrooms": [bedrooms],
    "Area Population": [population]
    })

    prediction = model.predict(house_data)
    print("\n===== Prediction Result =====")
    print(f"Predicted House Price: ${prediction[0]:,.2f}")

def show_menu():
    print("\n=== House Price Prediction ===")
    print("1. Show Dataset Information")
    print("2. Train Model")
    print("3. Predict House Price")
    print("4. Exit")



def main():
    df = load_data("data/USA_Housing.csv")
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42 )

    model = None
    while True:
        show_menu()
        choice = input("\nSelect option: ")
        if choice == "1":
            perform_eda(df)
        elif choice == "2":
            model = train_linear_regression(
                X_train,
                X_test,
                y_train,
                y_test)
            y_pred = model.predict(X_test)
            print("\nLinear Regression Results")
            print_metrics(y_test, y_pred)

        elif choice == "3":
            if model is None:
                print("\nPlease train the model first.")
            else:
                predict_house_price(model)

        elif choice == "4":
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid input.")

if __name__ == "__main__":
    if sys.stdin.isatty():
        main()