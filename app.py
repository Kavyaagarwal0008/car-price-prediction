from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load models and scaler
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form['model_choice']

    # Numerical inputs
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])
    tax = int(request.form['tax'])
    mpg = float(request.form['mpg'])
    engineSize = float(request.form['engineSize'])

    # Categorical inputs
    car_model = request.form['car_model']
    transmission = request.form['transmission']
    fuelType = request.form['fuelType']

    input_df = pd.DataFrame([[year, mileage, tax, mpg, engineSize,
                              car_model, transmission, fuelType]],
                            columns=['year','mileage','tax','mpg','engineSize',
                                     'model','transmission','fuelType'])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Choose model
    if selected_model == "model1":
        model = model1
        model_name = "Model 1 (Linear / Ridge)"
    else:
        model = model2
        model_name = "Model 2 (Lasso / Advanced)"

    # Align columns
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Scaling
    numeric_cols = ['year','mileage','tax','mpg','engineSize']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]

    return render_template(
        'index.html',
        prediction_text=f'{model_name} Predicted Price: ₹{int(prediction)}'
    )

if __name__ == "__main__":
    app.run(debug=True)
