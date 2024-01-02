# -*- coding: utf-8 -*-
"""flight_predictor.ipynb

Original file is located at
    https://colab.research.google.com/drive/1dJFmRyDsZtTj5VsoV2sdPduHsD8zXWLq

### Importing necessary packages !
"""

import pandas as pd
import numpy as np

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns

"""### Reading the dataset"""

train_data = pd.read_excel(r"Data_Train.xlsx")

"""### Exploratory Data Analysis[EDA]"""

train_data.head()

train_data.tail()

train_data.describe()

train_data.info()

train_data.isnull().sum()

train_data.shape

train_data.dtypes

train_data['Duration'].value_counts()

train_data.dropna(inplace = True) # Dropping the null values

train_data.isnull().sum() # After dropping the values

"""#### TO CONVERT SOME COLUMNS INTO DATE TIME FORMATS
##### From description we can see that Date_of_Journey is a object data type,
Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction,because our model will not be able to understand these string values,it just understand Time-stamp For this we require pandas to_datetime to convert object data type to datetime dtype.
"""

train_data.dtypes

train_data['Journey_day'] = pd.to_datetime(train_data.Date_of_Journey, format='%d/%m/%Y').dt.day

train_data['Journey_month'] = pd.to_datetime(train_data.Date_of_Journey, format='%d/%m/%Y').dt.month

train_data.head()

# Since we have converted Date_of_Journey column into integers, Now we can drop as it is of no use.

train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Departure time is when a plane leaves the gate.
# Similar to Date_of_Journey we can extract values from Dep_Time

# Extracting Hours
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour

# Extracting Minutes
train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute

# Now we can drop Dep_Time as it is of no use
train_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time

# Extracting Hours
train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour

# Extracting Minutes
train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

# Now we can drop Arrival_Time as it is of no use
train_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding duration_hours and duration_mins list to train_data dataframe

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins

train_data.drop(["Duration"], axis = 1, inplace = True)

train_data.head()

#### Converting the flight Dep_Time into proper time i.e. mid_night, morning, afternoon and evening.

def flight_dep_time(x):
    '''
    This function takes the flight Departure time
    and convert into appropriate format.

    '''

    if (x>4) and (x<=8):
        return "Early Morning"

    elif (x>8) and (x<=12):
        return "Morning"

    elif (x>12) and (x<=16):
        return "Noon"

    elif (x>16) and (x<=20):
        return "Evening"

    elif (x>20) and (x<=24):
        return "Night"

    else:
        return "late night"

train_data.columns

train_data['Duration_hours'].apply(flight_dep_time).value_counts().plot(kind="bar" , color="g")

"""### Applying pre-processing on duration column,
    -->> Once we pre-processed our Duration feature , lets extract Duration hours and minute from duration..
    
    -->> We have to tell our ML Model that this is hour & this is minute for each of the row ..
"""

train_data.head(3)

cat_cols = [col for col in train_data.columns if train_data[col].dtype == 'O'] # Categorical Columns
cat_cols

cont_cols = [col for col in train_data.columns if train_data[col].dtype != 'O'] # Continuous Columns
cont_cols

"""### Correlation Matrix Heatmap"""

selected_columns = ['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info','Price','Journey_day','Journey_month','Dep_hour','Dep_min',
                    'Arrival_hour','Arrival_min','Duration_hours','Duration_mins']

# Subset the DataFrame with selected columns
selected_df = train_data[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_df.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

"""## FEATURE ENCODING

#### Nominal data --> data are not in any order --> OneHotEncoder is used in this case
   #### Ordinal data --> data are in order --> LabelEncoder is used in this case
"""

categorical = train_data[cat_cols]
categorical.head(3)

categorical['Airline'].value_counts()

plt.figure(figsize=(15,5))
sns.boxplot(x='Airline' , y='Price' , data=train_data.sort_values('Price' , ascending=False))

"""It proves Jet Airways has the highest Price whereas All other Airlines have almost similar Prices"""

plt.figure(figsize=(15,5))
sns.boxplot(x='Total_Stops' , y='Price' , data=train_data.sort_values('Price' , ascending=False))

train_data["Airline"].value_counts()

# From graph we can see that Jet Airways Business have the highest Price.
# Apart from the first Airline almost all are having similar median

# Airline vs Price
sns.catplot(y = "Price", x = "Airline", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()

"""### Applying One-HotEncoding on data.."""

# As Airline is Nominal Categorical data we will perform OneHotEncoding

Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)
Airline = Airline.astype(int)

Airline.head()

train_data["Source"].value_counts()

# Source vs Price

sns.catplot(y = "Price", x = "Source", data = train_data.sort_values("Price", ascending = False),
            kind="boxen", height = 4, aspect = 3)
plt.show()

# As Source is Nominal Categorical data we will perform OneHotEncoding

Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)
Source = Source.astype(int)

Source.head()

train_data["Destination"].value_counts()

# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)
Destination = Destination.astype(int)
Destination.head()

train_data.head()
# train_data["Route"]

train_data["Route"]

"""### Dropping Unnecessary Columns"""

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other

train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

train_data["Total_Stops"].value_counts()

"""### Applying LabelEncoding on data.."""

# As this is case of Ordinal Categorical type we perform LabelEncoder
# Here Values are assigned with corresponding keys

train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4},
                   inplace = True)

train_data.head()

# Concatenate dataframe --> train_data + Airline + Source + Destination

data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)

data_train.head()

"""### Dropping Redundent Columns"""

data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

data_train.head()

data_train.shape

train_data.isnull().sum()

train_data.columns

categorical.isnull().sum()

"""### Total Categorical Data"""

for i in train_data.columns:
    print('{} has total {} categories'.format(i,len(train_data[i].value_counts())))

"""USING LABEL ENCODING"""

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

train_data.columns

train_data.head()

# extending the column limit
pd.set_option('display.max_columns', 35)
train_data.head()

"""#### Lets Perform outlier detection !

"""

def plot(df,col):
    fig,(ax1,ax2) = plt.subplots(2,1)
    sns.histplot(df[col],ax=ax1) # Distribution plot
    sns.boxplot(df[col],ax=ax2) # Box plot

plot(train_data , 'Price')

"""### Dealing with Outliers"""

# if prices greater than 35000 then they will be a outlier and replace them with median

train_data['Price']= np.where(train_data['Price'] > 35000,
                              train_data['Price'].median(), train_data['Price'])

plot(train_data , 'Price')

"""#### PERFORMING FEATURE SELECTION
###### Feature Selection:
    Finding out the best feature which will contribute and
    have good relation with target variable.
    It is used here To select important features OR to get rid
    of curse of dimensionality OR to get rid of duplicate features
"""

from sklearn.feature_selection import mutual_info_regression

train_data = train_data.fillna(0)

data_train.shape

data_train.columns

X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()

# X = train_data.drop(['Price'] , axis=1)
# X.head()

# y = data_train.iloc[:, 1]
y = train_data['Price']
y.head()

print(X.dtypes)

print(y.dtypes)

imp = mutual_info_regression(X , y)

"""#### Estimate mutual information for a continuous target variable.

###### Mutual information between two random variables is a non-negative value, which measures the dependency between the variables.
###### If It is equal to zero it means two random variables are independent, and higher values mean higher dependency.


"""

imp

imp_df = pd.DataFrame(imp , index=X.columns)

imp_df.columns = ['importance']

imp_df

imp_df.sort_values(by='importance' , ascending=False)

"""## Building the ML model"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.30, random_state=42)

"""### Creating a function predict() for predicting the accuracy and various scores of the model"""

from sklearn import metrics
def predict(ml_model):
    model = ml_model.fit(X_train, y_train)  # fitting the model
    print('Training score : {}'.format(model.score(X_train, y_train)))  # training score
    y_pred = model.predict(X_test)
    print('Predictions are : {}'.format(y_pred))  # prints the predicted values on the test data.
    print('\n')

    r2_score = model.score(X_test, y_test)  # corrected to calculate R2 score on the test data
    print('R2 score is : {}'.format(r2_score))  # prints the R2 score on the test data.

    print('MAE :', metrics.mean_absolute_error(y_test, y_pred))
    # MAE is the sum of absolute differences between our target and predicted variables.
    print('MSE :', metrics.mean_squared_error(y_test, y_pred))
    # MSE is the sum of squared differences between our target and predicted variables.
    print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # RMSE is the square root of the mean of the squared differences between our target and predicted variables.

    sns.displot(y_test - y_pred)
    plt.show()

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.show()

"""#### Fitting model using Linear Regression ALGO

"""

from sklearn.linear_model import LinearRegression

predict(LinearRegression())

"""#### Fitting model using Random Forest ALGO

"""

from sklearn.ensemble import RandomForestRegressor

predict(RandomForestRegressor())

"""#### Fitting model using K-Nearest Neighbors(KNN) ALGO

"""

from sklearn.neighbors import KNeighborsRegressor

predict(KNeighborsRegressor())

"""#### Fitting model using DECISION TREE ALGO

"""

from sklearn.tree import DecisionTreeRegressor

predict(DecisionTreeRegressor())

"""### PERFORMANCE COMPARISON OF THE ABOVE ML MODELS"""

LinearRegression = LinearRegression()
RandomForestRegressor = RandomForestRegressor()
KNeighborsRegressor = KNeighborsRegressor()
DecisionTreeRegressor = DecisionTreeRegressor()

import numpy as np
import matplotlib.pyplot as plt

# Your data
accuracy_scores = [0.6241794859094292, 0.9543104572465809, 0.7306851009410837, 0.9707490055980877]
r2_scores = [0.6241794859094292, 0.9543104572465809, 0.7306851009410837, 0.9707490055980877]
mae_scores = [1949.458356115105, 1163.7422094714325, 1873.2053912392364, 1361.270185947835]
mse_scores = [7835152.949901845, 4144050.3137799203, 8751130.411591165, 6020463.490129997]
rmse_scores = [2799.1343215183233, 2035.694061930702, 2958.2309598121587, 2453.6632796962986]

# List of algorithms
algorithms = ['Linear Regression', 'Random Forest', 'K-Neighbors', 'Decision Tree']

# Set the position of each bar on X-axis
r = np.arange(len(algorithms))

# Create figure and axis objects
fig, ax1 = plt.subplots(figsize=(12, 7))

# Create bar plot for MAE, MSE, and RMSE on the primary y-axis
bar_width = 0.2
ax1.bar(r - bar_width, mae_scores, width=bar_width, label='MAE', color='#1f78b4', edgecolor='black', linewidth=1.2)
ax1.bar(r, mse_scores, width=bar_width, label='MSE', color='#33a02c', edgecolor='black', linewidth=1.2)
ax1.bar(r + bar_width, rmse_scores, width=bar_width, label='RMSE', color='#e31a1c', edgecolor='black', linewidth=1.2)

# Set labels and ticks for the primary y-axis
ax1.set_xlabel('Algorithms', fontweight='bold', fontsize=12)
ax1.set_ylabel('Scores (Thousands)', fontweight='bold', fontsize=12)
ax1.set_xticks(r)
ax1.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

# Create a secondary y-axis for accuracy and R2 scores
ax2 = ax1.twinx()
line_accuracy, = ax2.plot(r - 2 * bar_width, np.array(accuracy_scores) * 100, marker='o', label='Precision', color='#fdbf6f', linewidth=2)
line_r2, = ax2.plot(r - bar_width, np.array(r2_scores) * 100, marker='o', label='R2 Score', color='#a6cee3', linewidth=2)

# Set labels and ticks for the secondary y-axis
ax2.set_ylabel('Scores (%)', fontweight='bold', color='#fdbf6f', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#fdbf6f')
ax2.set_ylim(0, 100)  # Assuming percentage scores are in the range [0, 1]

# Set logarithmic scale for the primary y-axis
ax1.set_yscale('log')

# Title
plt.title('Performance Comparison of ML Algorithms', fontweight='bold', fontsize=14)

# Adding a grid for better readability
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add legends for the lines
ax2.legend(handles=[line_accuracy, line_r2], loc='upper left', bbox_to_anchor=(1, 0.85), fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()