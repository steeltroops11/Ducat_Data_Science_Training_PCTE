# Heart Attack Risk Prediction

A Decision Tree Classifier project that predicts heart attack risk based on patient health and lifestyle data, with an interactive Streamlit web app for live predictions.

## Project Structure
```
Heart_Risk_Prediction/
├── notebooks/
│   └── Decision_Tree_Classifier.ipynb   # Data cleaning, EDA, model training & tuning
├── app.py                                # Streamlit web app (loads the saved model)
├── dt_model.pkl                          # Trained Decision Tree model
├── label_encoder.pkl                     # Saved LabelEncoder for the 'Sex' column
├── heart_attack_prediction_dataset.csv   # Dataset used for training
├── requirements.txt
└── README.md
```

## Dataset
The dataset contains 8,763 patient records with demographic, lifestyle, and clinical features (age, cholesterol, blood pressure, BMI, smoking, diet, etc.), with a binary target column `Heart Attack Risk` (0 = No Risk, 1 = Risk).

## Approach
1. **Preprocessing** — dropped irrelevant/high-cardinality columns (`Patient ID`, `Country`, `Continent`, `Hemisphere`), split `Blood Pressure` into `Systolic_BP`/`Diastolic_BP`, encoded categorical columns (`Sex`, `Diet`).
2. **Modeling** — trained a `DecisionTreeClassifier`, tuned hyperparameters with `GridSearchCV` (optimizing F1-score due to class imbalance), and compared `class_weight='balanced'` vs. SMOTE oversampling for handling the imbalanced target.
3. **Evaluation** — assessed with accuracy, precision, recall, F1-score, a confusion matrix heatmap, and feature importance plots.
4. **Deployment** — the final trained model and encoder were saved with `pickle` and served through a Streamlit app for interactive predictions.

## Running the App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Make sure `app.py`, `dt_model.pkl`, and `label_encoder.pkl` are all in the same directory.

## Notes
This project uses a public/synthetic dataset for academic purposes. Model predictions are not medical advice and the dataset's features were found to have a limited relationship with the target, which is discussed in the project notebook/report.
