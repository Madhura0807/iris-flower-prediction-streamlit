import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
st.title("Iris Classification with Streamlit")
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Get slider values
sepal_length = st.slider('Sepal Length', 0.0, 10.0, 5.0, 0.1)
sepal_width = st.slider('Sepal Width', 0.0, 10.0, 3.0, 0.1)
petal_length = st.slider('Petal Length', 0.0, 10.0, 1.0, 0.1)
petal_width = st.slider('Petal Width', 0.0, 10.0, 0.5, 0.1)

input_data = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = model.predict(input_data)

predicted_species = target_names[prediction[0]]

st.write(f"Predicted Species: {predicted_species}")