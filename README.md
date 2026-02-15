🚗 Ford Car Price Prediction (Machine Learning Regression)

Predict the resale price of Ford cars using machine learning based on real dealership vehicle specifications.

This project builds and evaluates regression models on real-world automobile data and compares different encoding strategies for optimal prediction performance.

📌 Problem Statement

Used car prices vary depending on multiple factors such as:

Model

Year

Transmission

Fuel Type

Mileage

Engine Size

MPG

Tax

Manually estimating resale price is unreliable and subjective.

This project builds a machine learning model that automatically predicts the fair market price of a Ford car.

📊 Dataset

Source: Kaggle — Ford Car Price Prediction
Records: 17,966 vehicles
Target Variable: price

Features
Feature	Description
model	Car model
year	Manufacturing year
transmission	Manual / Automatic / Semi-Auto
mileage	Distance driven
fuelType	Petrol / Diesel / Hybrid / Electric
tax	Road tax
mpg	Fuel efficiency
engineSize	Engine capacity
price	Car selling price (Target)
🧠 Project Workflow
1. Exploratory Data Analysis (EDA)

Performed detailed analysis:

Price distribution visualization

Correlation heatmap

Mileage vs Price relationship

Engine Size vs Price impact

Fuel Type price comparison

Transmission price comparison

Model wise price distribution

2. Data Preprocessing

Two different encoding pipelines were tested:

Pipeline A — One Hot Encoding
Categorical Encoding → One Hot Encoding
Feature Scaling → StandardScaler
Model → Linear Regression

Pipeline B — Label Encoding
Categorical Encoding → LabelEncoder
Feature Scaling → StandardScaler
Model → Linear Regression

3. Train Test Split
test_size = 0.33
random_state = 42

4. Model Used

Linear Regression

Reason:

Dataset relationships mostly linear

Baseline regression benchmark

High interpretability

📈 Model Evaluation

Metrics Used:

R² Score

Adjusted R² Score

Mean Absolute Error

Mean Squared Error

Adjusted R² was calculated manually to penalize unnecessary features.

🔬 Key Learning From Experiment

Comparing encoding strategies helps determine whether:

Treating categorical values as independent (One-Hot)
OR

Treating categories as ordinal (Label Encoding)

produces better generalization.

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn

👨‍💻 Author

Kamran Kausher
B.Tech CSE | AI/ML Enthusiast