# Car Price Prediction 

Car price prediction made using machine learning concepts and linear
Regression model in which large data set is given as input and depending
on various parameters the car price is predicted

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install python
pip install numpy
pip install pandas
pip install sklearn
pip install matplotlib
```

## The solution is divided into the following sections:

- Data understanding and exploration
- Data cleaning
- Data preparation
- Model building and evaluation
- Making predictions 

## Data Understanding and Exploration
Summary of data: 205 rows, 26 columns, no null values

![](https://raw.githubusercontent.com/Aniketsonii/Car-Price-Prediction/main/images/data%20understanding.png)

## Data Cleaning
We’ve seen that there are no missing values in the dataset.
Please make sure to update tests as appropriate.

![](https://raw.githubusercontent.com/Aniketsonii/Car-Price-Prediction/main/images/Data%20Cleaning.png)

## Data Preparation
Let’s now prepare the data for model building.

Split the data into X and y.

```bash
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(auto, train_size = 0.7, test_size = 0.3, random_state = 100)
y_train = df_train.pop('price')
X_train = df_train
```
## Model Building and Evaluation

```bash
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)
```
## Making Predictions 

```bash
num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']

df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')
X_test = df_test
X_test_new = X_test[['carwidth', 'horsepower', 'Luxury', 'hatchback']]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)
