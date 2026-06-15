from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Models load karo
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form se values lo
        features = [
            float(request.form['year']),
            float(request.form['gdp']),
            float(request.form['population']),
            float(request.form['energy_per_capita']),
            float(request.form['coal_co2']),
            float(request.form['oil_co2']),
            float(request.form['gas_co2']),
            float(request.form['methane']),
            float(request.form['primary_energy_consumption'])
        ]

        # Scale karo
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Predict karo
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        result = "🔴 High Emitter" if prediction == 1 else "🟢 Low Emitter"
        confidence = round(max(probability) * 100, 2)

        return render_template('index.html', 
                             prediction=result,
                             confidence=confidence)
    except Exception as e:
        return render_template('index.html', 
                             prediction=f"Error: {str(e)}",
                             confidence=0)

if __name__ == '__main__':
    app.run(debug=True)