import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def getNoOfCalls(data_call_pd):

    # create a dataframe with columns named as 'nd_name', 'uptime' and 'count'
    data_to_return = pd.DataFrame(columns=['nd_name', 'up_time', 'count'])

    # for each nd_name in data_call
    for nd_name in data_call_pd['nd_name'].unique():
        date_range = pd.date_range(start='2023-01-01', end='2023-01-31')
        temp_pd = pd.DataFrame(date_range, columns=['up_time'])
        temp_pd['nd_name'] = nd_name
        temp_pd['count'] = 0
        frames = [data_to_return, temp_pd]
        data_to_return = pd.concat(frames)

    frames = [data_to_return, data_call_pd]
    data_to_return = pd.concat(frames)

    # group by nd_name and up_time, get sum of count
    data_to_return = data_to_return.groupby(['nd_name', pd.Grouper(key='up_time', freq='H')])['count'].sum().reset_index(name='sum_count')

    # group  by node_name and log_date frequency D and create a list of sum_count
    _list = data_to_return.groupby(['nd_name', pd.Grouper(key='up_time', freq='D')])['sum_count'] \
        .agg(lambda x: list(x)).reset_index(name='list_call_no')
    data_to_return = pd.DataFrame(_list, columns=['node_name', 'up_time', 'list_call_no'])

    data_to_return['list_call_no'] = data_to_return['list_call_no'].apply(lambda x: np.array(x)).to_numpy()

    matrix = node_data_frame['list_call_no']

    # series with column t-1, t-2, t-3, t
    daily_call_number = pd.DataFrame(columns=['t-6', 't-5', 't-4', 't-3', 't-2', 't-1', 't'])

    for j in range(0, np.size(matrix)):
        # pad the array with zeros  to make it 24 hours
        matrix[j] = np.pad(matrix[j], (0, 24 - np.size(matrix[j])), 'constant', constant_values=(0, 0))
        daily_call_number.loc[j] = [get_call_in_day(matrix, j-6), get_call_in_day(matrix, j-5),
                          get_call_in_day(matrix, j-4), get_call_in_day(matrix, j-3),
                          get_call_in_day(matrix, j-2),
                          get_call_in_day(matrix, j-1), get_call_in_day(matrix, j)]

     matrix = np.vstack(matrix)

     # softmax normalization
     matrix = softmax(np.float64(matrix))

     # save the matrix as a csv file
     np.savetxt("Data/CallDataMatrix_" + nd_name + ".csv",
                       matrix,
                       delimiter=",")

     # save the daily_call_number as a csv file
     daily_call_number.to_csv(
                "CallPrediction/Data/DailyCallNumber_" + nd_name + ".csv", index=False)


    #write to csv
    data_to_return.to_csv('/Users/ezgi-lab/MLPipeline/data/atasehir_input.csv', index=False)

    print(data_to_return)

    return data_to_return




# Step 1: Data Collection
data_call = pd.read_csv('/Users/ezgi-lab/MLPipeline/data/atasehir.csv')

# Step 2: Data Preprocessing
# Assuming the dataset is already clean, we skip this step in this example.

# Step 3: Feature Engineering
data = getNoOfCalls(data_call)

# # Step 4: Model Selection and Training
# # Splitting the dataset into features (X) and target variable (y)
# X = data.drop('t_count', axis=1)
# y = data['t_count']
#
# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initializing and training the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
#
# # # Step 1: Data Collection
# # data_call = pd.read_csv('/Users/ezgi-lab/MLPipeline/data/atasehir.csv')
# #
# # # Step 2: Data Preprocessing
# # # Assuming the dataset is already clean, we skip this step in this example.
# #
# # # Step 3: Feature Engineering
# # data = getNoOfCalls(data_call)
# #
# # # Step 4: Model Selection and Training
# # # Splitting the dataset into features (X) and target variable (y)
# # X = data.drop('t_count', axis=1)
# # y = data['t_count']
# #
# # # Splitting the dataset into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # # Initializing and training the linear regression model
# # model = LinearRegression()
# # model.fit(X_train, y_train)
# #
# # # Step 5: Model Evaluation
# # # Making predictions on the testing set
# # y_pred = model.predict(X_test)
# #
# # # Creating a dataframe of actual and predicted values
# # df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# # print(df)
# # # visualizing the differences between actual and predicted values
# # df.plot(kind='bar', figsize=(10, 8))
# # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# # plt.show()
# #
# # # Calculating evaluation metrics
# # mse = mean_squared_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)
# # print("Mean Squared Error:", mse)
# # print("R-squared Score:", r2)
# #
# # # Step 6: Model Optimization
# # # Skipping this step in this basic example.
# #
# # # Step 7: Model Deployment
# # # Skipping this step in this basic example.
# #
# # # Step 8: Model Monitoring and Maintenance
# # # Skipping this step in this basic example.
