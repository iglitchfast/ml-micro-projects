import joblib

Weight_model = joblib.load('weight_model.pkl')
result = Weight_model.predict([[30,0,1,165]])

print("Predicted Weight for sample input:", result[0])
print("Model loaded and prediction made successfully.")