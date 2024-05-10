from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
import pickle
from pydantic import BaseModel
import xgboost as xgb

# Define the FastAPI app
app = FastAPI()

# Load the saved model and scaler from pickle files
with open("Final_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a Pydantic model for the input data
class InputData(BaseModel):
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: int
    Torque: float
    Tool_wear: int
    H: int
    L: int
    M: int

# Define an endpoint for making predictions
@app.post("/predict")
async def predict(input_data: InputData):
    # Convert input data to a numpy array in the correct order
    input_array = np.array([
        input_data.Air_temperature,
        input_data.Process_temperature,
        input_data.Rotational_speed,
        input_data.Torque,
        input_data.Tool_wear,
        input_data.H,
        input_data.L,
        input_data.M
    ]).reshape(1, -1)

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_array)

    # Create an XGBoost DMatrix for the model prediction
    dmatrix = xgb.DMatrix(scaled_input)

    # Use the model to make a prediction
    prediction = model.predict(dmatrix)

    # Return the prediction as a JSON response
    return {"prediction": prediction.tolist()}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
