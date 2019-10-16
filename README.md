
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
from sklearn.preprocessing import OneHotEncoder
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pandas as pd 
import numpy as np 
```

## Create Dataframe

The play tennis dataset is available in the repo as `tennis.csv`.  For this step, we'll start by importing the csv file as a pandas dataframe.


```python
# Load the dataset
df = pd.read_csv('tennis.csv')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outlook</th>
      <th>temp</th>
      <th>humidity</th>
      <th>windy</th>
      <th>play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sunny</td>
      <td>hot</td>
      <td>high</td>
      <td>False</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sunny</td>
      <td>hot</td>
      <td>high</td>
      <td>True</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>overcast</td>
      <td>hot</td>
      <td>high</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rainy</td>
      <td>mild</td>
      <td>high</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rainy</td>
      <td>cool</td>
      <td>normal</td>
      <td>False</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>



## Create Test and Training sets

Before we do anything we'll want to split our data into **_training_** and **_testing_** sets.  We'll accomplish this by first splitting the dataframe into features (`X`) and target (`y`), then passing `X` and `y` to the `train_test_split` function to create a 70/30 train test split. 


```python
X = df.loc[:, ['outlook', 'temp', 'humidity', 'windy']]
y = df.loc[:, 'play']

X_train, X_test , y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```

## Encode Categorical Data as numbers

Since all of our data is currently categorical (recall that each column is in string format), we need to encode them as numbers. For this, we'll use a handy helper object from sklearn's `preprocessing` module called `OneHotEncoder`.


```python
#One hot encode the training data and show the resulting dataframe with proper column names
ohe = OneHotEncoder()

ohe.fit(X_train)
X_train_ohe = ohe.transform(X_train).toarray()

#Creating this dataframe is not necessary its only to show the result of the ohe
ohe_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names(X_train.columns))

ohe_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outlook_overcast</th>
      <th>outlook_rainy</th>
      <th>outlook_sunny</th>
      <th>temp_cool</th>
      <th>temp_hot</th>
      <th>temp_mild</th>
      <th>humidity_high</th>
      <th>humidity_normal</th>
      <th>windy_False</th>
      <th>windy_True</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Train the Decision Tree 

One awesome feature of scikit-learn is the uniformity of its interfaces for every classifier--no matter what classifier we're using, we can expect it to have the same important methods such as `.fit()` and `.predict()`. This means that this next part will probably feel a little familiar.

We'll first create an instance of the classifier with any parameter values, and then we'll fit our data to the model using `.fit()` and make predictions with `X_test` using `.predict()`. 


```python
#Create the classifier, fit it on the training data and make predictions on the test set
clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train_ohe,y_train)

X_test_ohe = ohe.transform(X_test)
y_preds = clf.predict(X_test_ohe)
```

## Evaluate the Predictive Performance

Now that we have a trained model and we've generated some predictions, we can go on and see how accurate our predictions are. We can use a simple accuracy measure, AUC, a Confusion matrix, or all of them. This step is performed in the exactly the same manner , doesn't matter which  classifier you are dealing with. 

## Summary 

In this lesson, we looked at how to grow a decision tree in scikit-learn and python. We looked at different stages of data processing, training and evaluation that you would normally come across while growing a tree or training any other such classifier. We shall now move to a lab, where you will be required to build a tree for a given problem, following the steps shown in this lesson. 
