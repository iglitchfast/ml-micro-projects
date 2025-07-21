import pandas as pd
import matplotlib.pyplot as mt
import numpy as np  
from prettytable import PrettyTable
data = pd.read_csv('data.csv')
table = PrettyTable()
table.field_names = data.columns.tolist()

for row in data.values:
    table.add_row(row)

print(table)
table.field_names = data.columns.tolist()

x = data['height'].values
y = data['weight'].values
z = int(input("Enter a height to predict weight: "))
mt.scatter(x, y)
mt.xlabel('Height')
mt.ylabel('Weight')
mt.title('Height vs Weight')
mt.show()

m, c = np.polyfit(x, y, 1)
results_table = PrettyTable()
results_table.field_names = ["Description", "Value"]

equation = f"y = {m}x + {c}"
results_table.add_row(["THE MODEL IS: ", equation])
y_pred = m * z + c
results_table.add_row(["Predicted Weight (Y)", y_pred])

print(results_table)

