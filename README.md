# 🚗 Car Price Prediction System

A complete **Machine Learning based Web Application** that predicts the price of a car based on user inputs such as year, mileage, tax, MPG, engine size, fuel type, and transmission. The project includes model training, evaluation, and deployment using **Flask** and **Render**.

---

## 🔗 Live Demo

👉 **Live Application:** [https://car-price-prediction-wsk.onrender.com](https://car-price-prediction-wsk.onrender.com)

---

## 📌 Features

* Predict car prices using trained ML models
* Multiple models supported (Linear / Ridge / Lasso)
* User-friendly web interface
* Flask backend with HTML + CSS frontend
* Scikit-learn based ML pipeline
* Deployed on Render

---

## 🛠️ Tech Stack

### Programming & Libraries

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* Pickle

### Frontend

* HTML5
* CSS3

### Deployment & Tools

* Git & GitHub
* Render (Cloud Deployment)

---

## 📂 Project Structure

```
car-price-prediction/
│
├── app.py                # Flask application
├── train_model.py        # Model training script
├── streamlit_app.py      # (Optional) Streamlit version
├── requirements.txt      # Dependencies
├── Procfile              # Render deployment config
├── car_data.csv          # Dataset
├── model.pkl             # Trained ML model
├── model2.pkl            # Alternative model
├── scaler.pkl            # Feature scaler
│
├── templates/
│   └── index.html        # Main UI
│
├── static/
│   └── style.css         # Styling
│
└── README.md
```

---

## ⚙️ How It Works

1. User enters car details in the web form
2. Data is preprocessed using a saved scaler
3. Selected ML model predicts the price
4. Result is displayed on the UI

---

## ▶️ Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Kavyaagarwal0008/car-price-prediction.git
cd car-price-prediction
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

Visit: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📊 Machine Learning Models Used

* Linear Regression
* Ridge Regression
* Lasso Regression

Models are trained using historical car price data and evaluated before deployment.

---

## 📈 Future Improvements

* Add more ML models (Random Forest, XGBoost)
* Improve UI design
* Add input validation & error handling
* Show prediction confidence

---

## 👩‍💻 Author

**Kavya Agarwal**
B.Tech Student | Machine Learning Enthusiast

🔗 GitHub: [https://github.com/Kavyaagarwal0008](https://github.com/Kavyaagarwal0008)

---

## ⭐ Acknowledgements

* Scikit-learn documentation
* Render deployment guides
* Open-source datasets

---

⭐ If you like this project, don’t forget to **star the repository**!
