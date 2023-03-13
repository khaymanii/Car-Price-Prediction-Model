
# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# Data Collection and Processing

car_dataset = pd.read_csv('car data.csv')
car_dataset.head()
car_dataset.shape
car_dataset.info()


# Checking for missing values

print(car_dataset.isnull().sum())


# Checking the distribution of categorical data

print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())


# Categorical data encoding

car_dataset.replace({'Fuel_Type': { 'Petrol': 0, 'Diesel':1, 'CNG':2 }}, inplace=True)
car_dataset.replace({'Seller_Type': { 'Dealer': 0, 'Individual':1 }}, inplace=True)
car_dataset.replace({'Transmission': { 'Manual': 0, 'Automatic':1 }}, inplace=True)


print(car_dataset.head())

# Data splitting into Feature data and Target data

X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis = 1)
y = car_dataset['Selling_Price']

print(X)
print(y)


# Data splitting into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 2)


# Model Training : Linear Rgression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)


# Model evaluation : Training data

training_data_prediction = lin_reg.predict(X_train)

# Metrics using R squared eror

error_score = metrics.r2_score(y_train, training_data_prediction)
print('R squared Error : ', error_score)


# Visualize the Actual prices and Predicted prices

plt.scatter(y_train, training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs Predicted Prices')
plt.show()


# Model evaluation : Testing data

test_data_prediction = lin_reg.predict(X_test)


# Metrics using R squared eror in test data

error_score = metrics.r2_score(y_test, test_data_prediction)
print('R squared Error : ', error_score)


# Visualize the Actual prices and Predicted prices in test data

plt.scatter(y_test, test_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs Predicted Prices')
plt.show()


# Model Training : Lasso Rgression

lass_reg = Lasso()

lass_reg.fit(X_train, y_train)


# Model evaluation : Training data

training_data_prediction = lass_reg.predict(X_train)


# Metrics using R squared eror

error_score = metrics.r2_score(y_train, training_data_prediction)
print('R squared Error : ', error_score)


# Visualize the Actual prices and Predicted prices

plt.scatter(y_train, training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs Predicted Prices')
plt.show()


# Model evaluation : Testing data

test_data_prediction = lass_reg.predict(X_test)


# Metrics using R squared eror in test data

error_score = metrics.r2_score(y_test, test_data_prediction)
print('R squared Error : ', error_score)


# Visualize the Actual prices and Predicted prices in test data

plt.scatter(y_test, test_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

