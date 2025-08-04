import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

age = int(input("Enter Age: "))
gender = input("Enter Gender: ")
bodytype = input("Enter Body Type: ")
height = int(input("Enter Height: "))

mydata = pd.read_csv("dataset_weight_finder.csv")

le_gender = LabelEncoder()
le_body = LabelEncoder()
mydata['Gender_encoded'] = le_gender.fit_transform(mydata['Gender'])
mydata['BodyType_encoded'] = le_body.fit_transform(mydata['BodyType'])

x = mydata[['Age', 'Gender_encoded', 'BodyType_encoded', 'Height']]
y = mydata[['Weight']]

model = LinearRegression()
model.fit(x, y)

gender_encoded = le_gender.transform([gender])[0]
bodytype_encoded = le_body.transform([bodytype])[0]
user_input = [[age, gender_encoded, bodytype_encoded, height]]

predicted_value = model.predict(user_input)
print("Predicted Weight:", predicted_value[0][0])

y_pred_all = model.predict(x)
score = r2_score(y, y_pred_all)
print("Accuracy Score : ", score)