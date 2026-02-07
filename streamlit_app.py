import streamlit as st
import pickle
import pandas as pd

model1 = pickle.load(open('model1.pkl','rb'))
model2 = pickle.load(open('model2.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

st.title("🚗 Car Price Prediction")

model_choice = st.selectbox(
    "Select Model",
    ("Model 1 (Linear/Ridge)", "Model 2 (Lasso/Advanced)")
)

year = st.number_input("Year", 2000, 2025)
mileage = st.number_input("Mileage")
tax = st.number_input("Tax")
mpg = st.number_input("MPG")
engineSize = st.number_input("Engine Size")

car_model = st.text_input("Car Model")
transmission = st.selectbox("Transmission", ["Manual","Automatic","Semi-Auto"])
fuelType = st.selectbox("Fuel Type", ["Petrol","Diesel","Hybrid"])

if st.button("Predict"):
    df = pd.DataFrame([[year,mileage,tax,mpg,engineSize,
                        car_model,transmission,fuelType]],
        columns=['year','mileage','tax','mpg','engineSize',
                 'model','transmission','fuelType'])

    df = pd.get_dummies(df)

    model = model1 if model_choice.startswith("Model 1") else model2
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    numeric_cols = ['year','mileage','tax','mpg','engineSize']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    price = model.predict(df)[0]
    st.success(f"Predicted Price: ₹{int(price)}")
