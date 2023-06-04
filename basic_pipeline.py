import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Collection
data_call = pd.read_csv('/Users/ezgi-lab/MLPipeline/data/atasehir.csv')

# Step 2: Data Preprocessing
# Assuming the dataset is already clean, we skip this step in this example.

# Step 3: Feature Engineering
# incoming file has columns named as 'uptime', 'nd_name' and 'count'
# we need to create dateset with columns named as 't_7_count', 't_1_count', 't_count'
# from incoming file as intermediate step
data = pd.read_csv('dataset.csv')

# Step 4: Model Selection and Training
# Splitting the dataset into features (X) and target variable (y)
X = data.drop('t_count', axis=1)
y = data['t_count']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Making predictions on the testing set
y_pred = model.predict(X_test)

# Creating a dataframe of actual and predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
# visualizing the differences between actual and predicted values
df.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


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
