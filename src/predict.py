import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

def predict_revenue(temp):
    temp = np.array([[temp]])
    return model.predict(temp)[0]

if __name__ == "__main__":
    t = float(input("Enter temperature: "))
    print("Predicted Revenue =", predict_revenue(t))
