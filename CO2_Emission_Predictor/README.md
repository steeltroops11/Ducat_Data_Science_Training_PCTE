# 🌍 CO2 Emission Predictor

A Machine Learning web application that predicts whether a country is a **High or Low CO2 Emitter** based on key economic and energy indicators.

---

## 🎯 Project Overview

| Detail | Info |
|--------|------|
| **Dataset** | CO2 & Greenhouse Gas Emissions (Our World in Data) |
| **Rows** | 14,366 |
| **Features** | 80 |
| **Best Model** | Random Forest |
| **Accuracy** | 98.68% |

---

## 🤖 Models Compared

| Model | Accuracy |
|-------|----------|
| Linear Regression | R² = 0.459 |
| Logistic Regression | 89.74% |
| Decision Tree | 96.49% |
| **Random Forest** | **98.68% 🏆** |

---

## 📊 Key Insights

- **Energy per capita** is the most important feature (27%)
- **Coal CO2** is the biggest polluter (19%)
- **Year** is the least important factor (3%) — emissions depend on energy use, not time!

---

## 🛠️ Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```
Then open → `http://127.0.0.1:5000`

---

## 📁 Project Structure
CO2_Emission_Predictor/

│

├── app.py

├── rf_model.pkl

├── scaler.pkl

├── requirements.txt

├── README.md

└── templates/

└── index.html

---

## 👨‍💻 Developed By

**Navish** | Data Science Trainee @ Ducat PCTE                   