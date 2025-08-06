import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("car_data.csv")

BrandEncoder = LabelEncoder()
FuelEncoder = LabelEncoder()

df["Brand_enc"] = BrandEncoder.fit_transform(df["Brand"])
df["Fuel_enc"] = FuelEncoder.fit_transform(df["FuelType"])

X = df[["Year", "Brand_enc", "Fuel_enc", "Mileage"]]
Y = df[["Price"]]

print(X.head())
print(Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(Dense(10, activation="relu", input_shape=(4,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, Y_train, epochs=5, batch_size=10)
regressor = LinearRegression()
regressor.fit(X, Y)

joblib.dump(BrandEncoder, "BrandEncoder.pkl")
joblib.dump(FuelEncoder, "FuelEncoder.pkl")
joblib.dump(regressor, "regressor.pkl")
model.save("CarPriceModel.keras")
print("Training complete and all files saved.")