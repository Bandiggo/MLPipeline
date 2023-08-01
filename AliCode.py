import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#takes a matrix and a day as input and returns the sum of calls made on that day.
def get_call_in_day(matrix, day):
    if day < 0: return 0;
    return np.sum(matrix[day])

#takes a pandas DataFrame data_call_pd as input and returns a new DataFrame data_to_return that contains the number of calls made by each node at each hour of the day.
def getNoOfCalls(data_call_pd):
    # create a dataframe with columns named as 'nd_name', 'uptime' and 'count'
    data_to_return = pd.DataFrame(columns=['nd_name', 'up_time', 'count'])

    counter = 0
   # for each nd_name in data_call
    for nd_name in data_to_return['nd_name'].unique():
            # date format is 2022-12-01 00:00:00
            # create a date range from 2022-12-01 00:00:00 to 2023-02-28 23:00:00 with 1 hour frequency
            # closed = left means that the first date is included in the range
            date_range = pd.date_range(start='2022-12-01 00:00:00', end='2023-02-28 23:00:00', freq='H', closed='left')
            temp_pd = pd.DataFrame(date_range, columns=['up_time'])
            temp_pd['nd_name'] = nd_name
            temp_pd['count'] = 0
            frames = [data_to_return, temp_pd]
            data_to_return = pd.concat(frames)

    frames= [data_to_return, data_call_pd]
    data_to_return = pd.concat(frames) 

    # Convert 'up_time' column to Timestamp type
    data_to_return['up_time'] = pd.to_datetime(data_to_return['up_time'])

    # group by nd_name and up_time, get sum of count
    list_ = \
            data_to_return.groupby(['nd_name', pd.Grouper(key='up_time', freq='H')])['count'] \
                .sum().reset_index(name='sum_count')

    data_to_return = pd.DataFrame(list_, columns=['nd_name', 'up_time', 'sum_count'])


    # group by node_name and up_time frequency D and create a list of sum_count
    _list = data_to_return.groupby(['nd_name', pd.Grouper(key='up_time', freq='D')])['sum_count'] \
        .agg(lambda x: list(x)).reset_index(name='list_call_no')
    data_to_return = pd.DataFrame(_list, columns=['nd_name', 'up_time', 'list_call_no'])

    data_to_return['list_call_no'] = data_to_return['list_call_no'].apply(lambda x: np.array(x)).to_numpy()





    # for each node in data_frame
    for nd_name in data_to_return['nd_name'].unique():

            # get the data of a specific node
            node_data_frame = data_to_return[data_to_return['nd_name'] == nd_name].reset_index()


        
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
            # matrix = softmax(np.float64(matrix))

            # save the matrix as a csv file
            np.savetxt("CallDataMatrix_" + nd_name + ".csv",
                       matrix,
                       delimiter=",")



            # save the daily_call_number as a csv file
            daily_call_number.to_csv(
                "DailyCallNumber_" + nd_name + ".csv", index=False)
            
            #Execute multiple logistic regression, input daily_call_number columns except the last column which is 't', output is 't'
    #sucessfully executed
    #nw_gp_ol56_01.34est5, nw_gp_ol58_01.34ky18
    nd_name = 'nw_gp_ol58_01.34ky18'
    daily_call_number = pd.read_csv("DailyCallNumber_" + nd_name + ".csv")
    feature_cols = ['t-6', 't-5', 't-4', 't-3', 't-2', 't-1']
    X = daily_call_number[feature_cols]
    y = daily_call_number.t





    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a logistic regression model
    model = LogisticRegression(random_state=16)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('y test')
    print(y_test)
    print('y pred')
    print(y_pred)
    print('x test')
    print(X_test)
    print('x train')
    print(X_train)
    print('y train')
    print(y_train)
    #display the results
    print(f"Node: {nd_name}, Mean Squared Error: {mse}, R-squared: {r2}")
    #plot y_test and y_pred 
    plotMultipleVector(y_test, y_pred)


    return data_to_return


def plotMultipleVector(X, Y):
        x = np.array(range(0, 9))
        y = np.array(range(0, 9))
        plt.title("Plotting 1-D array")
        plt.plot(x, X, color="red", label="Array elements")
        plt.plot(y, Y, color="green", label="Array elements", linestyle='dashed')
        plt.scatter(x, X, color="red", label="Array elements")
        plt.scatter(y, Y, color="green", label="Array elements")
        plt.legend()
        plt.show()

# Step 1: Data Collection
data_call = pd.read_csv('data_frame.csv')

# Step 2: Data Preprocessing
# Assuming the dataset is already clean, we skip this step in this example.

# Step 3: Feature Engineering
data = getNoOfCalls(data_call)

