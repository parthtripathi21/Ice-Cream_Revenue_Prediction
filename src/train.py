import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_model():
    df = pd.read_csv("data/Ice_Cream.csv")

    X = df[['Temperature']]
    y = df['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=2529
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    pickle.dump(model, open("model/model.pkl", "wb"))

    print("Model Trained")

if __name__ == "__main__":
    train_model()
