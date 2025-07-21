import pandas as pd
import matplotlib.pyplot as mt
import sklearn as sk
from sklearn.linear_model import LinearRegression 
data = pd.read_csv('new_data.csv')

"""mt.scatter(data['height'], data['weight'])
mt.xlabel('Height')
mt.ylabel('Weight')
mt.title('Height vs Weight')
mt.show()"""

while True:
    try:
        data = pd.read_csv('new_data.csv')
        break
    except FileNotFoundError:
        print("\nFile not found. Please ensure 'new_data.csv' is in the correct directory.")
        break
while True:
    x = data[['height']].values
    y = data[['weight']].values
    if x.size == 0 or y.size == 0:
        print("\nData is empty. Please check 'new_data.csv'.")
        break
    
    #print("\n",data)

    model = LinearRegression()
    model.fit(x, y)
    z = int(input("\nEnter a height to predict weight: "))

    predval = model.predict([[z]])

    print("\n y = ",predval)