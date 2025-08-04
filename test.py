import pandas as pd
import numpy as np
import math
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
import sklearn.neighbors as ng
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
mydata=pd.read_csv("dataset_weight_finder.csv")

gn_enc=LabelEncoder()
body_enc=LabelEncoder()
mydata["gender_enc"]=gn_enc.fit_transform(mydata[["Gender"]])
mydata["body_enc"]=body_enc.fit_transform(mydata[["BodyType"]])
x=mydata[["Age","gender_enc","body_enc","Height"]]
y=mydata[["Weight"]]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
weight_model=ng.KNeighborsClassifier(n_neighbors=5)
weight_model.fit(x_train,y_train)
print("Training is completed")
joblib.dump(weight_model,"Weight_model.pkl")
test_result=weight_model.predict(x_test)
print("MSE",mean_squared_error(y_test,test_result))
print("RMSE",np.sqrt(mean_squared_error(y_test,test_result)))
print("R2_score=",r2_score(y_test,test_result))
print("ACCURACY SCORE",accuracy_score(y_test,test_result)*100)
print("CONFUSION MATRIX=",confusion_matrix(y_test,test_result))

weight_model=joblib.load("Weight_model.pkl")
result=weight_model.predict([[30,0,1,160]])
print("THE RESULT=",result[0])