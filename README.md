Machine Learning Model
import pandas as pd
from sklearn.linear_model import LinearRegression
import ipywidgets as widgets
import numpy as np
from IPython.display import display

file_path = "ozone.csv"
data = pd.read_csv(file_path)

data['Date_Num'] = pd.to_datetime(data['Date']).map(lambda x: (x - pd.Timestamp("2024-01-01")).days)
data['Time_Num'] = pd.to_datetime(data['Time'], format='%H:%M:%S').map(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

data['Ozone_Thickness'] = data['Temperature (°C)'] * 0.2 + data['Date_Num'] * 0.1 + 5 + np.random.normal(0, 2, len(data))

X = data[['Temperature (°C)', 'Date_Num']]
y = data['Ozone_Thickness']

model = LinearRegression()
model.fit(X, y)

def predict_ozone_thickness(temperature, date):
    date_num = (pd.Timestamp(date) - pd.Timestamp("2024-01-01")).days

    prediction = model.predict(pd.DataFrame([[temperature, date_num]], columns=['Temperature (°C)', 'Date_Num']))
    return f"Predicted Ozone Thickness using Nanosatellite datas by Ajay and prem bros: {prediction[0]:.2f} dobson units"

temperature_widget = widgets.FloatText(value=25.0, description='Temperature (°C):')
date_widget = widgets.Text(value='2024-12-10', description='Date (YYYY-MM-DD):')

output = widgets.Output()

def on_value_change(change):
    with output:
        print(predict_ozone_thickness(temperature_widget.value, date_widget.value))

temperature_widget.observe(on_value_change, names='value')
date_widget.observe(on_value_change, names='value')

display(temperature_widget, date_widget, output)
------------------------------------------------------------------------------------------------------------------------


Data Preprocessing and Visualizing





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dx=pd.read_csv("thickey.csv")
sns.violinplot(x="thickness_of_ozone",y="date",data=dx)
plt.title("Data Visualization done by prem and Ajay K")
plt.show()

![download](https://github.com/user-attachments/assets/4bd48948-ceed-46c4-9b60-9e7d192adfd4)
![download](https://github.com/user-attachments/assets/9bd3a076-d6e1-457e-b3b4-b62f9d66a9cf)


import seaborn as sns
sns.histplot(df['Temperature'])
plt.show()






