import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Dictionary to store results of different models
results = {}

# Load the training and testing datasets
x_train = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\x_test_processed.csv')
y_train = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\y_train.csv')
x_test = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\x_test_processed.csv')
y_test = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\y_test.csv')

# Parameter grids for different models
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

param_grid_knn = {
    'n_neighbors': [3, 5],
    'weights': ['uniform', 'distance']
}

param_grid_dt = {
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

param_grid_lgb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1]
}

# Dictionary of models and their corresponding parameter grids
models = {
    'RandomForestRegressor': (RandomForestRegressor(random_state=42), param_grid_rf),
    'KNeighborsRegressor': (KNeighborsRegressor(), param_grid_knn),
    'DecisionTreeRegressor': (DecisionTreeRegressor(random_state=42), param_grid_dt),
    'LightGBM': (LGBMRegressor(random_state=42), param_grid_lgb)
}

# Dictionary to store results of different models
results = {}

# Iterate over each model and perform RandomizedSearchCV
for model_name, (model, param_grid) in models.items():
    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                n_iter=10, cv=2, n_jobs=-1, random_state=42, verbose=1)
    search.fit(x_train, y_train.values.ravel())  # Fit the model
    best_model = search.best_estimator_  # Get the best model
    y_pred = best_model.predict(x_test)  # Predict on the test set
    
    # Store the results
    results[model_name] = {
        'R2': best_model.score(x_test, y_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Best Params': search.best_params_ 
    }

# Print the results for each model
for model_name, result in results.items():
    print(f'{model_name}:')
    print(f"  R²: {result['R2']:.4f}")
    print(f"  RMSE: {result['RMSE']:.4f}")
    print(f"  Best Parameters: {result['Best Params']}")
    print()
    
# Train the best model (RandomForestRegressor) with the best parameters found
best_model = RandomForestRegressor(min_samples_split=5, n_estimators=100, random_state=42, max_depth=20)
scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation RMSE: ", np.mean(np.sqrt(-scores)))

# Fit the best model on the training data
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)

# Print the final test R² and RMSE
print("Final Test R²: ", best_model.score(x_test, y_test))
print("Final Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plot feature importances
importances = best_model.feature_importances_
feature_names = x_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance - RandomForest")
sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="Blues_d")

# Save the plot
output_path = 'E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\reports\\figures\\'
plt.savefig(f'{output_path}top_5_feature_importance_random_forest.png', format='png', dpi=300, bbox_inches='tight')
