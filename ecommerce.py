
#for data analysis
import pandas as pd

#for plotting the necessary graphs
import matplotlib.pyplot as plt

#for linear algebra
import numpy as np
from sklearn.model_selection import train_test_split

#regression analysis
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#plotting graphs
import seaborn as sns

#loading and viewing dataset
data = pd.read_csv('C:/Users/Akash/Desktop/Ecommerce Customers.csv')
print(data.head(5))
print(data.info())
print(data.describe())

#plotting a graph for the parameters
#it shows graph between users using website vs mobile app.
#by this companies can decide where they have chances of profit
sns.pairplot(data,x_vars=['Time on App','Time on Website'],y_vars=['Yearly Amount Spent'],height=7,kind='reg')
plt.show()
sns.pairplot(data,x_vars=['Length of Membership'],y_vars=['Yearly Amount Spent'],height=7,kind='reg')
plt.show()

x = data[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y  = data['Yearly Amount Spent']

#training the data set 30% test and 70% train data
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=101)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
model = LinearRegression()
model.fit(X_train,Y_train)
print('Intercept : ',model.intercept_)
print('Coefficients: ',model.coef_)
y_pred = model.predict(X_test)

#scatterplor for predicted value of y
plt.scatter(Y_test,y_pred)
plt.xlabel('Test value of Y')
plt.ylabel('Predict value of y')
plt.show()

#other information for verification
print("Accuracy score = ",model.score(X_train,Y_train))
print('Root Mean Squared Error : ',np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))
print('Mean Squared Error : ',metrics.mean_squared_error(Y_test,y_pred))
print('Mean Absolute Error : ',metrics.mean_absolute_error(Y_test,y_pred))


