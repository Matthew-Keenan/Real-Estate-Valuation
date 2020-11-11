import numpy as np 
import pandas as pd 
import sklearn 
from sklearn import linear_model
from sklearn.metrics import r2_score

# Reads csv file into pandas dataframe, drops unnecessary column
data = pd.read_csv('Real estate valuation data set.csv', sep=',')
data = data.drop(['No'], 1)

# Seta house price per unit area as the value to predict 
value_to_predict = 'House Price Per Unit Area'

# Creates numpy array with all columns except the one being predicted
x = np.array(data.drop([value_to_predict], 1))

# x array values were set in scientific notation, set here to float values
y = np.array(data[value_to_predict])

# Splits x and y datasets: x array training values, x array testing values, y array training values, and y array testing values
# Separates 10% of each dataset to be used for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Initialize 'linear' variable to be linear regression object, trains 'linear' with the two training sets
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# Forms predictions based on testing set
y_predict = linear.predict(x_test)

# Calculates the R2 Score given y_test values and y_predict values
R2 = r2_score(y_test, y_predict)

# Neatly displays data found
print(f"R2 Value: {round(R2, 6)}")
print()
for i in range(len(y_predict) - 1):
    print(f"Predicted Value: {round(y_predict[i], 2)} \t Actual Given Value: {round(y_test[i], 2)}" )