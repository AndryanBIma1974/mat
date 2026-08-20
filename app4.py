# =========================================================
# ❤️ HEART DISEASE PREDICTION USING MLP
# STREAMLIT APP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =========================================================
# 1. PAGE
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Prediction")
st.write("MLP Neural Network")

# =========================================================
# 2. LOAD DATASET
# =========================================================

data = pd.read_csv("heart.csv")

st.success("✅ Dataset loaded successfully!")

with st.expander("View Dataset"):
    st.write("Dataset size:", data.shape)
    st.dataframe(data.head())

# =========================================================
# 3. CHECK DATA
# =========================================================

with st.expander("Dataset Information"):
    st.write("Columns:")
    st.write(data.columns.tolist())

    st.write("Missing Values:")
    st.write(data.isnull().sum())

# =========================================================
# 4. FEATURES AND TARGET
# =========================================================

X = data.drop("output", axis=1)
y = data["output"]

# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# 6. STANDARDIZATION
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 7. MLP MODEL
# =========================================================

model = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================================================
# 8. PREDICTION
# =========================================================

y_pred = model.predict(X_test_scaled)

# =========================================================
# 9. ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Training Samples", len(X_train))
col3.metric("Testing Samples", len(X_test))

# =========================================================
# 10. CLASSIFICATION REPORT
# =========================================================

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

st.dataframe(pd.DataFrame(report).transpose())

# =========================================================
# 11. CONFUSION MATRIX
# =========================================================

st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(ax=ax)

ax.set_title("Heart Disease Prediction - MLP")

st.pyplot(fig)

# =========================================================
# 12. SAMPLE PATIENT
# =========================================================

st.subheader("🩺 Sample Patient Prediction")

patient = X.iloc[0].values.reshape(1, -1)

patient_scaled = scaler.transform(patient)

prediction = model.predict(patient_scaled)[0]

probability = model.predict_proba(patient_scaled)[0]

if prediction == 1:
    st.warning("⚠️ Higher likelihood of heart disease")
else:
    st.success("✅ Lower likelihood of heart disease")

st.write(
    f"No disease: **{probability[0] * 100:.2f}%**"
)

st.write(
    f"Disease: **{probability[1] * 100:.2f}%**"
)

# =========================================================
# 13. ACTUAL VS PREDICTED
# =========================================================

st.subheader(" Actual vs Predicted")

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.dataframe(results.head(20))

st.success(" MLP Heart Disease Prediction Completed!")