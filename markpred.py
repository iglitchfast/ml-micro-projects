import pandas as pd
import matplotlib.pyplot as mt
import sklearn as sk
from sklearn.linear_model import LinearRegression
data = pd.read_csv('studydata.csv')
print("\n", data)

while True:
    x = data[['hour']].values
    y = data[['score']].values
    model = LinearRegression()
    model.fit(x, y)
    z = int(input("\nEnter hour to predict score: "))
    predval = model.predict([[z]])
    print("\nPredicted score for hour is:", predval)