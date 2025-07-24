import pandas as pd
# Removed unused imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv('salary.csv')
print("\n", data)

while True:
    
    
    x = data[['job']].values
    y = data[['salary']].values
    model = LinearRegression()
    encoder = OneHotEncoder()
    x_encoded = encoder.fit_transform(data[['job']])
    model.fit(x_encoded, y)
    z = input("\nEnter a job to predict salary[plustwo,btech]: ")
    z_encoded = encoder.transform([[z]]).toarray()
    predval = model.predict(z_encoded)
    print("\nPredicted salary for job is: ", predval)