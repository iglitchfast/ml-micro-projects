import pandas as pd
import matplotlib.pyplot as mt
import sklearn as sk
from sklearn.linear_model import LinearRegression
data = pd.read_csv('fuel.csv')
print("\n", data)

while True:
    x = data[['month']].values
    y = data[['price']].values
    model = LinearRegression()
    model.fit(x, y)
    z = int(input("\nEnter a month to predict fuel price: "))
    predval = model.predict([[z]])
    print("\nPredicted fuel price for month is:", predval)