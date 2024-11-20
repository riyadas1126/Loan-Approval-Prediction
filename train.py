# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Setting display options for better visibility
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Reading the training dataset
df = pd.read_csv("train.csv")

# Displaying the first few rows of the dataset
print(df.head())

# Basic data information and summary
print("Dataset Shape:", df.shape)
df.info()
print(df.describe())

# Checking missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Imputing missing values with appropriate strategies
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)

# Exploratory Data Analysis (EDA)

# Boxplot of ApplicantIncome to check outliers
df.boxplot(column="ApplicantIncome")
plt.title("Applicant Income Distribution")
plt.ylabel("Applicant Income")
plt.show()

# Histogram of ApplicantIncome
df["ApplicantIncome"].hist(bins=20)
plt.title("Applicant Income Histogram")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

# Boxplot and histogram of CoapplicantIncome
df.boxplot(column="CoapplicantIncome")
plt.title("Coapplicant Income Distribution")
plt.ylabel("Coapplicant Income")
plt.show()

df["CoapplicantIncome"].hist(bins=20)
plt.title("Coapplicant Income Histogram")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

# Creating a TotalIncome column (ApplicantIncome + CoapplicantIncome)
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

# Log transformation of TotalIncome
df["TotalIncome_log"] = np.log(df["TotalIncome"]).values
df["TotalIncome_log"].hist(bins=20)
plt.title("Log Transformed Total Income Histogram")
plt.xlabel("Log(Total Income)")
plt.ylabel("Frequency")
plt.show()

# Boxplot and histogram of LoanAmount
df.boxplot(column="LoanAmount")
plt.title("Loan Amount Distribution")
plt.ylabel("Loan Amount")
plt.show()

df["LoanAmount"].hist(bins=20)
plt.title("Loan Amount Histogram")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()

# Log transformation of LoanAmount
df["LoanAmount_log"] = np.log(df["LoanAmount"]).values
df["LoanAmount_log"].hist(bins=20)
plt.title("Log Transformed Loan Amount Histogram")
plt.xlabel("Log(Loan Amount)")
plt.ylabel("Frequency")
plt.show()

# Preparing data for modeling
x = df.iloc[:, np.r_[1:5, 9:11, 13:15]].values  # Selecting feature columns
y = df.iloc[:, 12].values  # Selecting target column

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Encoding categorical variables
LabelEncoder_x = LabelEncoder()
for i in range(0, 5):
    x_train[:, i] = LabelEncoder_x.fit_transform(x_train[:, i])
x_train[:, 7] = LabelEncoder_x.fit_transform(x_train[:, 7])

LabelEncoder_y = LabelEncoder()
y_train = LabelEncoder_y.fit_transform(y_train)

for i in range(0, 5):
    x_test[:, i] = LabelEncoder_x.fit_transform(x_test[:, i])
x_test[:, 7] = LabelEncoder_x.fit_transform(x_test[:, 7])
y_test = LabelEncoder_y.fit_transform(y_test)

# Standardizing the features
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# Modeling and Evaluating Algorithms

# Decision Tree Classifier
dtc_model = DecisionTreeClassifier()
dtc_model.fit(x_train, y_train)
dtc_pred = dtc_model.predict(x_test)
print("Decision Tree Accuracy:", accuracy_score(dtc_pred, y_test))

# Random Forest Classifier
rfc_model = RandomForestClassifier()
rfc_model.fit(x_train, y_train)
rfc_pred = rfc_model.predict(x_test)
print("Random Forest Accuracy:", accuracy_score(rfc_pred, y_test))

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)
accuracy_lr = accuracy_score(lr_pred, y_test)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Accuracy (percentage):", accuracy_lr * 100, "%")

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
print("Naive Bayes Accuracy:", accuracy_score(nb_pred, y_test))

# Preparing and Predicting on the Test Dataset
df1 = pd.read_csv("test.csv")

# Handling missing values in the test dataset
df1["Gender"].fillna(df1["Gender"].mode()[0], inplace=True)
df1["Dependents"].fillna(df1["Dependents"].mode()[0], inplace=True)
df1["Self_Employed"].fillna(df1["Self_Employed"].mode()[0], inplace=True)
df1["Loan_Amount_Term"].fillna(df1["Loan_Amount_Term"].mode()[0], inplace=True)
df1["Credit_History"].fillna(df1["Credit_History"].mode()[0], inplace=True)
df1["LoanAmount"].fillna(df1["LoanAmount"].mean(), inplace=True)

# Visualizing Loan Amount in the test data
df1.boxplot(column="LoanAmount")
plt.title("Test Data Loan Amount Distribution")
plt.ylabel("Loan Amount")
plt.show()

df1["LoanAmount"].hist(bins=20)
plt.title("Test Data Loan Amount Histogram")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()

# Log transforming LoanAmount in the test data
df1["LoanAmount_log"] = np.log(df1["LoanAmount"])
df1["LoanAmount_log"].hist(bins=20)
plt.title("Test Data Log Transformed Loan Amount Histogram")
plt.xlabel("Log(Loan Amount)")
plt.ylabel("Frequency")
plt.show()

# Calculating Total Income for test data
df1["TotalIncome"] = df1["ApplicantIncome"] + df1["CoapplicantIncome"]
df1["TotalIncome_log"] = np.log(df["TotalIncome"])

# Encoding test data
test = df1.iloc[:, np.r_[1:5, 9:11, 13:15]].values
for i in range(0, 5):
    test[:, i] = LabelEncoder_x.fit_transform(test[:, i])
test[:, 7] = LabelEncoder_x.fit_transform(test[:, 7])
test = ss.transform(test)

# Predicting using Naive Bayes model
pred = nb_model.predict(test)

# Adding predictions to the test data
df1["Loan_Status"] = pred
df1["Loan_Status"] = df1["Loan_Status"].map({1: "Yes", 0: "No"})

# Saving the test data with predictions
df1.to_csv("test_with_predictions.csv", index=False)
