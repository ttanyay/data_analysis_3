import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
import requests
from zipfile import ZipFile
import io

st.title("Extra Assignment - MSE Decomposition")
st.header("Course 'Prediction with Machine Learning for Economists 2024/25 Winter'")
st.header("Students: Ayazhan Toktargazy, Tatyana Yakushina")

st.write("Data Source: UCI Machine Learning Repository")

# Function to load dataset
def load_data():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    response = requests.get(data_url)
    zip_file = ZipFile(io.BytesIO(response.content))
    zip_file.extractall("./student_data")
    students = pd.read_csv("./student_data/student-mat.csv", sep=";")
    return students

# Load dataset
students = load_data()
X = students.drop("G3", axis=1)
y = students["G3"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    "Dummy Model (High Bias)": DummyRegressor(strategy="mean"),
    "Simple Linear (G1 Only)": LinearRegression(),
    "Moderate Model (G1, G2, Study time, Failures)": LinearRegression(),
    "Full Model (All Features)": LinearRegression()
}

features = {
    "Simple Linear (G1 Only)": ["G1"],
    "Moderate Model (G1, G2, Study time, Failures)": ["G1", "G2", "studytime", "failures"],
    "Full Model (All Features)": X_train.columns.tolist()
}

predictions = {}
mse = {}
bias = {}
variance = {}
bias_squared = {}

for name, model in models.items():
    if name in features:
        model.fit(X_train[features[name]], y_train)
        preds = model.predict(X_test[features[name]])
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    predictions[name] = preds
    mse[name] = mean_squared_error(y_test, preds)
    bias[name] = np.mean(preds) - np.mean(y_test)
    variance[name] = np.var(preds)
    bias_squared[name] = bias[name] ** 2

# Display Metrics
st.write("### Model Performance Metrics")
st.dataframe(pd.DataFrame({"Model": models.keys(), "MSE": mse.values(), "Bias": bias.values(), "Bias²": bias_squared.values(), "Variance": variance.values()}))

# Display all four figures together
st.write("### Model Predictions vs Actual G3")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, preds) in enumerate(predictions.items()):
    axes[idx].scatter(y_test, preds, color="purple", alpha=0.6)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Prediction")
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Actual G3")
    axes[idx].set_ylabel("Predicted G3")
    axes[idx].legend()

plt.tight_layout()
st.pyplot(fig)

# Bias-Variance Decomposition Plot
st.write("### Bias-Variance Tradeoff Visualization")
fig, ax1 = plt.subplots(figsize=(10,6))

x = np.arange(len(models))
width = 0.2

ax1.bar(x - width/2, list(variance.values()), width=width, label="Variance", color='orange')
ax1.bar(x + width/2, list(mse.values()), width=width, label="MSE", color='green')
ax1.set_xticks(x)
ax1.set_xticklabels(models.keys(), rotation=15)
ax1.set_ylabel("Variance / MSE")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(x, list(bias.values()), color='blue', marker='o', label="Bias")
ax2.set_ylabel("Bias")
ax2.legend(loc="upper right")

plt.title("Bias-Variance Decomposition Across Models")
st.pyplot(fig)


# Bias-Variance Decomposition Plot
st.write("### Bias-Variance Tradeoff Visualization")
fig, ax1 = plt.subplots(figsize=(10,6))

x = np.arange(len(models))
width = 0.2

ax1.bar(x - width, list(variance.values()), width=width, label="Variance", color='orange')
ax1.bar(x, list(mse.values()), width=width, label="MSE", color='green')
ax1.set_xticks(x)
ax1.set_xticklabels(models.keys(), rotation=15)
ax1.set_ylabel("Variance / MSE")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(x, list(bias_squared.values()), color='skyblue', marker='o', label="Bias²")
ax2.set_ylabel("Bias²")
ax2.legend(loc="upper right")

plt.title("Bias², Variance, and MSE Across Models")
st.pyplot(fig)

# Actual vs Predicted Plot for Each Model
st.write("### Actual vs Predicted for Each Model")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, preds) in enumerate(predictions.items()):
    axes[idx].scatter(y_test, preds, color="purple", alpha=0.6)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Prediction")
    axes[idx].set_title(f"{name}: Actual vs Predicted")
    axes[idx].set_xlabel("Actual G3")
    axes[idx].set_ylabel("Predicted G3")
    axes[idx].legend()

plt.tight_layout()
st.pyplot(fig)
