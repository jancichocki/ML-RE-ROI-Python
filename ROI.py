# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the data (Assuming you have a CSV file named 'real_estate_data.csv')
df = pd.read_csv('real_estate_data.csv')

# Define features (X) and target (y)
X = df.drop('ROI', axis=1)  # All columns except 'ROI'
y = df['ROI']  # Return on Investment column

# Data Preprocessing - normalize the data using Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Instantiate Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict ROI for the test set
y_pred = rf_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest MSE: {mse}")
