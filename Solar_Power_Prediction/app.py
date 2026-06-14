from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("solar_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    ambient = float(request.form["ambient"])
    module = float(request.form["module"])
    irradiation = float(request.form["irradiation"])

    data = pd.DataFrame({
        "AMBIENT_TEMPERATURE":[ambient],
        "MODULE_TEMPERATURE":[module],
        "IRRADIATION":[irradiation]
    })

    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction,2)
    )

if __name__ == "__main__":
    app.run(debug=True)