"""
Linear Regression Model

This script trains a linear regression model on the MPG dataset.
"""
# Import libraries
from datetime import datetime
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
mpg_df = sns.load_dataset('mpg')
print(mpg_df.head())

# Summary of the dataset
mpg_df.info()
print(mpg_df.isnull().sum())

# Drop the 'name' column
mpg_df.drop('name', axis=1, inplace=True)

# Summary of the dataset
mpg_df.info()
print(mpg_df.isnull().sum())

# Drop rows with missing values
mpg_df.dropna(inplace=True)

# Distribution of all variables
plt.figure(figsize=(16, 10))
for i, column in enumerate(mpg_df.columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(mpg_df[column], kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()

# Show the plot
plt.show()

# Select only the numeric columns for the correlation matrix
numeric_df = mpg_df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap to visualize the correlations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Drop 'acceleration' and 'displacement' to avoid multicollinearity
mpg_df.drop(["acceleration", "displacement"], axis=1, inplace=True)

# Create 'age' feature from 'model_year' and drop the 'model_year' column
this_year = datetime.today().year
mpg_df["age"] = this_year - mpg_df["model_year"]
mpg_df.drop(["model_year"], axis=1, inplace=True)

# Convert 'origin' to dummy variables with one-hot encoding
mpg_df = pd.get_dummies(mpg_df, columns=['origin'], drop_first=True)
mpg_df.head()

# Model building

# Define the features and target variable
X = mpg_df.drop("mpg", axis=1)
y = mpg_df["mpg"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression lm_model
lm_model = LinearRegression()

# Train the lm_model on the training data
lm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lm_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display the results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Prediction error plot
# !pip install yellowbrick

# Instantiate the visualizer
visualizer = PredictionError(lm_model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show() # Finalize and render the figure

# Instantiate the visualizer
visualizer = ResidualsPlot(lm_model)

visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
visualizer.show() # Finalize and render the figure

#TODO Improve the model

# Add polynomial and interaction features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

# Fit model with polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

# Evaluate the model
X_test_poly = poly.transform(X_test)
y_pred_poly = poly_model.predict(X_test_poly)

mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Features - MAE: {mae_poly}, RMSE: {rmse_poly}, R²: {r2_poly}")

# Ridge Regression
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate Ridge model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - MAE: {mae_ridge}, RMSE: {rmse_ridge}, R²: {r2_ridge}")

# Addressing Non-Linearity

# Apply log transformation
mpg_df['log_weight'] = np.log(mpg_df['weight'])
mpg_df['sqrt_horsepower'] = np.sqrt(mpg_df['horsepower'])

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - MAE: {mae_rf}, RMSE: {rmse_rf}, R²: {r2_rf}")

# Regularization

from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Metrics
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - MAE: {mae_ridge}, RMSE: {rmse_ridge}, R²: {r2_ridge}")

# Cross-Validation and Hyperparameter Tuning

from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
cv_scores = cross_val_score(ridge_model, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R² Scores:", cv_scores)

# Hyperparameter tuning
param_grid = {'alpha': [0.1, 1.0, 10]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
