
# Building Trees using scikit-learn

## Introduction

In this lesson, we shall cover decision trees (for classification) in python, using scikit-learn and pandas. The emphasis will be on the basics and understanding the resulting decision tree. Scikit-Learn provides a consistent interface for running different classifiers/regressors. For classification tasks, evaluation is performed using the same measures as we have seen before. Let's look at our example from earlier lessons and grow a tree to find our solution. 

## Objectives
You will be able to:

- Using `pandas` to prepare the data for the scikit-learn decision tree algorithm
- Train the classifier with a training dataset and evaluate performance using different measures
- Visualize the decision tree and interpret the visualization

## Import Necessary Libraries

In order to prepare data, train, evaluate and visualize a decision tree , we would need a number of packages in python. Here are the packages that you would normally consider importing before moving on. Run the cell below to import everything we'll need for this lesson. 


```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import tree 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import pandas as pd 
import numpy as np 
```

## Create Dataframe

The play tennis dataset is available in the repo as `tennis.csv`.  For this step, we'll start by importing the csv file as a pandas dataframe. Then, since all of our data is currently categorical (recall that each column is in string format), we need to encode them as numbers. For this, we'll use a handy helper objects from sklearn's `preprocessing` module. Since our target, `play`, is in a binary format, we'll use `LabelEncoder`. Since our predictors are not binary, we'll instead use `OneHotEncoder` for them. Finally, we'll print the shape of each piece of transformed data in order to make sure that it all looks correct. 
- Apply labels to target variable such that `yes=1` and `no=0`
- Apply one hot encoding to the feature set, creating ten features (outlook x 3, temp x 3, humidity x 2 , wind x 2) 
- Print the resulting features and check shape


```python
# Load the dataset
df = pd.read_csv('tennis.csv') 

# Create label encoder instance
lb = LabelEncoder() 

# Create Numerical labels for classes
df['play_'] = lb.fit_transform(df['play'] ) 
df['outlook_'] = lb.fit_transform(df['outlook']) 
df['temp_'] = lb.fit_transform(df['temp'] ) 
df['humidity_'] = lb.fit_transform(df['humidity'] ) 
df['windy_'] = lb.fit_transform(df['windy'] ) 

# Split features and target variable
X = df[['outlook_', 'temp_', 'humidity_', 'windy_']] 
Y = df['play_']

# Instantiate a one hot encoder
enc = OneHotEncoder()

# Fit the feature set X
enc.fit(X)

# Transform X to onehot array 
onehotX = enc.transform(X).toarray()

onehotX, onehotX.shape, X.shape
```

    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\preprocessing\_encoders.py:368: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
    If you want the future behaviour and silence this warning, you can specify "categories='auto'".
    In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
      warnings.warn(msg, FutureWarning)





    (array([[0., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
            [0., 0., 1., 0., 1., 0., 1., 0., 0., 1.],
            [1., 0., 0., 0., 1., 0., 1., 0., 1., 0.],
            [0., 1., 0., 0., 0., 1., 1., 0., 1., 0.],
            [0., 1., 0., 1., 0., 0., 0., 1., 1., 0.],
            [0., 1., 0., 1., 0., 0., 0., 1., 0., 1.],
            [1., 0., 0., 1., 0., 0., 0., 1., 0., 1.],
            [0., 0., 1., 0., 0., 1., 1., 0., 1., 0.],
            [0., 0., 1., 1., 0., 0., 0., 1., 1., 0.],
            [0., 1., 0., 0., 0., 1., 0., 1., 1., 0.],
            [0., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
            [1., 0., 0., 0., 0., 1., 1., 0., 0., 1.],
            [1., 0., 0., 0., 1., 0., 0., 1., 1., 0.],
            [0., 1., 0., 0., 0., 1., 1., 0., 0., 1.]]), (14, 10), (14, 4))



## Create Test and Training sets

Our data is now encoded properly, but we're still not ready for training. Before we do anything with a Decision Tree model, we'll want to split our data into **_training_** and **_testing_** sets.  We'll accomplish this by passing `onehotX` and `Y` to the `train_test_split` function to create a 70/30 train test split. 


```python
X_train, X_test , y_train,y_test = train_test_split(onehotX, Y, test_size = 0.3, random_state = 42) 
```

## Train the Decision Tree 

One awesome feature of scikit-learn is the uniformity of its interfaces for every classifier--no matter what classifier we're using, we can expect it to have the same important methods such as `.fit()` and `.predict()`. This means that this next part will probably feel a little familiar.

We'll first create an instance of the classifier with any parameter values, and then we'll fit our data to the model using `.fit()` and make predictions with `X_test` using `.predict()`. 


```python
clf= DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train) 
y_pred = clf.predict(X_test)
```

## Evaluate the Predictive Performance

Now that we have a trained model and we've generated some predictions, we can go on and see how accurate our predictions are. We can use a simple accuracy measure, AUC, a Confusion matrix, or all of them. This step is performed in the exactly the same manner , doesn't matter which  classifier you are dealing with. 

## Summary 

In this lesson, we looked at how to grow a decision tree in scikit-learn and python. We looked at different stages of data processing, training and evaluation that you would normally come across while growing a tree or training any other such classifier. We shall now move to a lab, where you will be required to build a tree for a given problem, following the steps shown in this lesson. 
