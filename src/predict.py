import joblib
import numpy as np

def load_model(path):
    return joblib.load(path)

def predict(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]