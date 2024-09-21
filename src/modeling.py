import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
import seaborn as sns
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Dictionary to store the results of different models
results = {}

# Load the training and testing datasets
x_train = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\x_test_processed.csv')
y_train = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\y_train.csv')
x_test = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\x_test_processed.csv')
y_test = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\y_test.csv')

# Train and evaluate Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results['LinearRegression'] = (model.score(x_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred)))

# Train and evaluate Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train.values)
y_pred = model.predict(x_test)
results['RandomForestRegressor'] = (model.score(x_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred)))

# Train and evaluate K-Neighbors Regressor model
model = KNeighborsRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results['KNeighborsRegressor'] = (model.score(x_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred)))

# Train and evaluate Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results['Decision Tree'] = (model.score(x_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred)))

# Train and evaluate Gradient Boosting (LightGBM) model
model = LGBMRegressor(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results['LightGBM'] = (model.score(x_test, y_test), np.sqrt(mean_squared_error(y_test, y_pred)))
results['LightGBM']

# Print the results of all models
for key, value in results.items():
    print(f'{key} : {value}')
    
# Set the style for seaborn plots
sns.set_style("whitegrid")

# Extract model names, R^2 scores, and RMSE values from the results dictionary
models = list(results.keys())
r2_scores = [value[0] for value in results.values()]
rmse_values = [value[1] for value in results.values()]

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})

# Define color palettes for the bars
r2_colors = sns.color_palette("Blues", len(models))
rmse_colors = sns.color_palette("Reds", len(models))

# Plot horizontal bar charts for R^2 scores and RMSE values
bars1 = ax1.barh(models, r2_scores, color=r2_colors, edgecolor='black')
ax1.set_xlabel('R^2 Score', fontsize=12)
ax1.set_title('Model R² Scores', fontsize=14)
ax1.grid(True, axis='x', linestyle='--', alpha=0.7)

bars2 = ax2.barh(models, rmse_values, color=rmse_colors, edgecolor='black')
ax2.set_xlabel('RMSE', fontsize=12)
ax2.set_title('Model RMSE', fontsize=14)
ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add text annotations to the bars in the R^2 score plot
for bar in bars1:
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, 
             f'{bar.get_width():.4f}', va='center', ha='left', fontsize=10, color='black')

# Add text annotations to the bars in the RMSE plot
for bar in bars2:
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, 
             f'{bar.get_width():.4f}', va='center', ha='left', fontsize=10, color='black')

# Adjust layout and save the figure to a file
plt.tight_layout(pad=3.0)
output_path = 'E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\reports\\figures\\model_performance.png'
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
