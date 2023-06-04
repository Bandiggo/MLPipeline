import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Collection
data = pd.read_csv('dataset.csv')

# Step 2: Data Preprocessing
# Assuming the dataset is already clean, we skip this step in this example.

# Step 3: Feature Engineering
# Assuming the dataset doesn't require additional feature engineering, we skip this step in this example.

# Step 4: Model Selection and Training
# Splitting the dataset into features (X) and target variable (y)
X = data.drop('target_variable', axis=1)
y = data['target_variable']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Making predictions on the testing set
y_pred = model.predict(X_test)

# Calculating evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Step 6: Model Optimization
# Skipping this step in this basic example.

# Step 7: Model Deployment
# Skipping this step in this basic example.

# Step 8: Model Monitoring and Maintenance
# Skipping this step in this basic example.