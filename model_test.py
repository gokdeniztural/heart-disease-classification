import pickle
import pandas as pd

def main():
    #load model
    with open("heart_model.pkl", "rb") as f:
        pipeline = pickle.load(f)


    X_test = pd.read_csv("test_data.csv")
    print(pipeline.predict(X_test))


if __name__ == "__main__":
    main()


