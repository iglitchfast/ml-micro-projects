import joblib
import numpy as np
from tensorflow.keras.models import load_model

BrandEncoder = joblib.load("BrandEncoder.pkl")
FuelEncoder = joblib.load("FuelEncoder.pkl")
regressor = joblib.load("regressor.pkl")
model = load_model("CarPriceModel.keras")

print("Car Price Prediction System")
print("Type 'exit' as brand name to stop.\n")

while True:
    
    year = int(input("Enter year of the car (2016): "))
    brand = input("Enter brand name (Maruti): ")
    fuel = input("Enter fuel type (Petrol/Diesel/Electric/CNG): ")
    mileage = float(input("Enter mileage (24.0): "))
    
    brand_enc = BrandEncoder.transform([brand])[0]
    fuel_enc = FuelEncoder.transform([fuel])[0]

    input_data = [[year, brand_enc, fuel_enc, mileage]]

    price_pred = regressor.predict(input_data)
    print("Predicted Price (Linear Regression): ₹", price_pred[0][0])

    nn_pred = model.predict(np.array(input_data), verbose=0)
    print("Predicted Price (Neural Network): ₹", nn_pred[0][0])

    print("-" * 40)

'''I have toggled the code to make it 
print both linear regression value and neural network value.
I have also added a loop to allow multiple predictions for 
ease of retrying values for myself.'''