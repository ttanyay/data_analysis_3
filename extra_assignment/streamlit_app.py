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

st.write("This assignment investigates the bias-variance tradeoff using both simulated and real-world data. The goal is to analyze how different modeling complexities impact predictive accuracy, bias, and variance.")

st.title("Part 1: Simulation Data")

st.write("In this section, we generate synthetic data to explore the effects of model complexity on prediction performance. We analyze how bias and variance change as we adjust the sample size and model type.")

# Here we have an option to choose a sample size (100, 500, 1000), and also comparison of MSE, Var, Bias on the graph

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

st.title("Bias-Variance Tradeoff Simulation: Academic performance based on previous grades, chocolate and other factors📊")

# Sidebar for sample size selection
sample_size = st.sidebar.selectbox("Select Sample Size", [100, 500, 1000], index=0)

# Simulate data
np.random.seed(42)
n_students = sample_size

# Academic performance
G1 = np.random.randint(5, 20, n_students)
G2 = G1 + np.random.randint(-3, 4, n_students)

# Additional features
study_time = np.random.randint(1, 15, n_students)  # hours/week
chocolates_from_Gabor = np.random.randint(0, 5, n_students)  # motivational chocolates 🍫
attendance = np.random.randint(50, 101, n_students)  # % attendance
sleep_hours = np.random.uniform(5, 9, n_students)  # hours/night
internet_usage = np.random.uniform(1, 6, n_students)  # non-study hours/day

# Final grade G3 influenced by all features
G3 = (
    0.25 * G1 +
    0.3 * G2 +
    0.2 * study_time +
    0.5 * chocolates_from_Gabor +
    0.1 * attendance / 10 -
    0.2 * internet_usage +
    0.1 * sleep_hours +
    np.random.randn(n_students) * 2  # random noise
).round()

# Clip G3 between 0 and 20
G3 = np.clip(G3, 0, 20)

# Create DataFrame
data = pd.DataFrame({
    'G1': G1,
    'G2': G2,
    'Study Time': study_time,
    'Chocolates from Gabor': chocolates_from_Gabor,
    'Attendance (%)': attendance,
    'Sleep Hours': sleep_hours,
    'Internet Usage': internet_usage,
    'Final Grade (G3)': G3
})

# Show sample data
st.write(data.head())

# Visualization: Study Time vs Final Grade
fig, ax = plt.subplots()
scatter = ax.scatter(data['Study Time'], data['Final Grade (G3)'], c=data['Attendance (%)'], cmap='viridis', alpha=0.7)
ax.set_xlabel("Study Time (hours/week)")
ax.set_ylabel("Final Grade (G3)")
ax.set_title("Study Time vs Final Grade (colored by Attendance)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Attendance (%)')
st.pyplot(fig)

# Define features and target
X = data.drop(columns=['Final Grade (G3)'])
y = data['Final Grade (G3)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),
    "Simple Linear Regression": LinearRegression(),
    "Polynomial Regression (Degree 2)": LinearRegression()
}

# Train and evaluate
results = []
model_predictions = {}
for name, model in models.items():
    if name == "Polynomial Regression (Degree 2)":
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    bias = np.mean(y_pred) - np.mean(y_test)
    variance = np.var(y_pred)
    results.append({"Model": name, "Sample Size": n_students, "MSE": mse, "Bias": bias, "Variance": variance})
    model_predictions[name] = y_pred

# Show results
st.write("### Bias-Variance Metrics Across Sample Sizes")
results_df = pd.DataFrame(results)
st.dataframe(results_df)

# Plot Bias-Variance Comparison with dual y-axis and histograms
st.write("### Bias-Variance Comparison")
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

width = 0.2  # Bar width
x = np.arange(len(results_df['Model'].unique()))

colors = {'MSE': 'skyblue', 'Variance': 'salmon'}

# Plot MSE and Variance with separate colors
for idx, metric in enumerate(['MSE', 'Variance']):
    values = [results_df[results_df['Model'] == model][metric].values[0] for model in results_df['Model'].unique()]
    ax1.bar(x + idx * width, values, width=width, label=metric, color=colors[metric])

# Plot Bias on secondary y-axis
bias_values = [results_df[results_df['Model'] == model]['Bias'].values[0] for model in results_df['Model'].unique()]
ax2.plot(x + width, bias_values, color='black', marker='o', label='Bias')

ax1.set_xlabel("Model")
ax1.set_ylabel("MSE / Variance")
ax2.set_ylabel("Bias")
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(results_df['Model'].unique())
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_title("Bias-Variance Tradeoff Across Models")
st.pyplot(fig)

# Interactive model selection for plotting
st.write("### Model Predictions")
selected_model = st.selectbox("Select Model to View Predictions", list(models.keys()))

fig, ax = plt.subplots()
ax.scatter(y_test, y_test, label="Actual", color="blue", alpha=0.5)
y_pred = model_predictions[selected_model]
ax.scatter(y_test, y_pred, label=selected_model, alpha=0.6)
ax.set_xlabel("Actual Grades")
ax.set_ylabel("Predicted Grades")
ax.legend()
st.pyplot(fig)


st.write("The first model using the dummy regressor represents high bias, considering no data patterns, and 0 variance, because predictions do not change, being stable.")
st.write("The second model, simple linear regression model, is a usual model with moderate bias and variance.")
st.write("The third model, polynomial regression, represents how with higher complexity bias is reduced but increases variance.")
st.write("We have also experimented with sample size, simulating data on 100, 500, and 1000 students.")


st.title("Part 2: Real data")
st.write("Data Source: UCI Machine Learning Repository") 

st.write("The real world data consists of student records from two Portuguese secondary schools and contains information about demographics, social factors, and academic performance.")
st.write("We are predicting G3 (final grade) based on various student features, including G1 (first period grade), G2 (second period grade), study time, failures, and more.")

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

# Additional Plot
st.write("### Model Predictions for Each Model with Metrics")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, preds in enumerate(predictions.values()):
    axes[idx].scatter(X_test["G1"], y_test, color="blue", alpha=0.5, label="Actual G3")
    axes[idx].scatter(X_test["G1"], preds, color="red", alpha=0.5, label="Predicted G3")
    if idx < 2:
        axes[idx].plot(X_test["G1"], preds, color="red", alpha=0.7, linewidth=1)
    axes[idx].set_title(f"{list(models.keys())[idx]} Model\nBias: {bias[list(models.keys())[idx]]:.2f}, Variance: {variance[list(models.keys())[idx]]:.2f}, MSE: {mse[list(models.keys())[idx]]:.2f}")
    axes[idx].set_xlabel("G1 (First Period Grade)")
    axes[idx].set_ylabel("G3 (Final Grade)")
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

st.write("As model complexity increases from the Dummy Regressor (predicting the mean) to using G1, then G1, G2, study time, failures, and finally all explanatory variables, bias decreases as the model captures more data patterns, while variance increases due to higher sensitivity to fluctuations in the training data—illustrating the classic bias-variance tradeoff.")
