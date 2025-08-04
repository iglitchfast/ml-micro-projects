import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LinearRegression
import joblib

df=pd.read_csv("persons.csv")

GenderEncoder=LabelEncoder()
BodyTypeEncoder = LabelEncoder()

df["Gender_enc"]=GenderEncoder.fit_transform(df["Gender"])
df["BodyType_enc"]=BodyTypeEncoder.fit_transform(df["BodyType"])

X = df[["Age","Gender_enc","BodyType_enc","Height"]]
Y = df[["Weight"]]

print(X)
print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model=Sequential()
model.add(Dense(10,activation="relu",input_shape=(4,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mean_squared_error")

model.fit(X_train,Y_train,epochs=5,batch_size=10)

model=LinearRegression()
model.fit(X,Y)

print(model.predict([[25,0,0,170]]))