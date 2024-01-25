# -*- coding: utf-8 -*-
"""
Linear Regression (Baseline)

Jorge Martinez-Gil: A Comparative Study of Ensemble Techniques Based on Genetic Programming: A Case Study in Semantic Similarity Assessment. Int. J. Softw. Eng. Knowl. Eng. 33(2): 289-312 (2023)

@author: Jorge Martinez-Gil
"""

import scipy.stats
import pandas as pd
import numpy as np

def my_pearson(x, y):
    """
    Calculates the Pearson correlation coefficient between two arrays.

    Parameters:
    x (numpy.ndarray): The first array.
    y (numpy.ndarray): The second array.

    Returns:
    float: The Pearson correlation coefficient between x and y.
    """
    a = x.flatten()
    b = y.flatten()
    return -1*np.corrcoef(a, b)[0, 1]
    
def my_spearman(x, y):
    """
    Calculates the Spearman correlation coefficient between two arrays.

    Parameters:
    x (numpy.ndarray): The first array.
    y (numpy.ndarray): The second array.

    Returns:
    float: The Spearman correlation coefficient between x and y.
    """
    a = x.flatten()
    b = y.flatten()
    rho, p = scipy.stats.spearmanr(x, y)
    return -1*rho
    
# Load the training data
input_file = "mc-training.txt"
categories = ['b', 'c', 'd', 'e', 'f']
raw_dataset = pd.read_csv(input_file, error_bad_lines=False) 
df = pd.read_csv(input_file, skipinitialspace=True, usecols=categories) 
y = pd.DataFrame(raw_dataset, columns=['a']) 

# Split the training data into input and output arrays
X_train = df.to_numpy()
y_train = y.to_numpy()

# Load the validation data
input_file2 = "mc-validation.txt"
categories2 = ['bert-cos', 'bert-inn', 'bert-man', 'bert-euc', 'bert-ang']
raw_dataset = pd.read_csv(input_file2, skipinitialspace=True, delimiter=',', error_bad_lines=False) 
df = pd.read_csv(input_file2, skipinitialspace=True, usecols=categories2)
y = pd.DataFrame(raw_dataset, columns=['truth'])

# Split the validation data into input and output arrays
X_test = df.to_numpy()
y_test = y.to_numpy()

# Train a linear regression model on the training data
reg = LinearRegression().fit(X_train, y_train)

# Use the trained model to make predictions on the validation data
y_pred = reg.predict(X_test)

# Print the predicted and actual values for the validation data
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

# Calculate and print the Pearson and Spearman correlation coefficients between the predicted and actual values
print(my_pearson(y_test, y_pred))
print(my_spearman(y_test, y_pred))