The
Basics:

# Statistics in Python:
# numpy is the package that is best suited to make calculations about data
import numpy as np

data = [15, 16, 18, 19, 22, 24, 29, 30, 34]
import numpy as np

data = [15, 16, 18, 19, 22, 24, 29, 30, 34]

print("mean:", np.mean(data))  # Calculates mean of the data
print("median:", np.median(data))  # Calculates median of the data
print("50th percentile (median):", np.percentile(data, 50))  # Calculates 50th percentile of the data
print("25th percentile:", np.percentile(data, 25))  # Calculates 25th percentile of the data
print("75th percentile:", np.percentile(data, 75))  # Calculates 75th percentile of the data
print("standard deviation:", np.std(data))  # Calculates standard deviation of the data
print("variance:", np.var(data))  # Calculates the variance of the data

# Reading Data With Pandas:
# Pandas is best suited for viewing and manipulating data
import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
"""
df gives the data frame that we are creating and pd.read_csv('') sets the data frame to the
data in parentheses. df.head() shows the first five lines of the data frame.
"""
print(df.head())

pd.options.display.max_columns = 6
"""
pd.options.display.max_columns = 6 forces python to display six columns, when it may have otherwise
displayed more or less. pd.read... just sets our data frame and df.describe shows a numerical breakdown
of each of the columns of data.
"""
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.describe())

# Manipulating Data with Pandas
col = df['Fare']
"""
Selects a single columns and creates a new data frame of just that column
"""
print(col)

small_df = df[['Age', 'Sex', 'Survived']]
"""
You can do the same thing with a few columns rather than just one. This can be useful for organizing
at data you think should be part of a regression.
"""
print(small_df.head())

df['male'] = df['Sex'] == 'male'
"""
You can create a column based on a boolean of a different column. The male column is a boolean
based off the gender column.
"""
print(df['male'].head())

# Numpy Basics:

df['Fare']
df['Fare'].values
"""
Adding ".values" gives you the specified data as a numpy array
"""

df[['Pclass', 'Fare', 'Age']].values
"""
".values" also works with multiple columns
"""

arr = df[['Pclass', 'Fare', 'Age']].values
"""
If you create an array of values, you can see the shape of that array with ".shape"
"""
print(arr.shape)

arr[0, 1]
print(arr[0])
print(arr[:, 2])
"""
The first line of code will give you a specific row and specific column of one data piece
The second line will give you the entire row that you specify
The last line will give you the value of the second column for all 887 rows
"""

# More with Numpy Arrays:
mask = arr[:, 2] < 18
"""
This creates an array of boolean values. In this case if the person on the Titanic is over
or under the age of 18 (adult or child).
"""
arr[mask]
"""
This will then take the previously defined array but only show the rows in the array for
which the boolean value is true (only show the rows the are children).
"""
arr[arr[:, 2] < 18]
"""
You don't have to create the mask and can instead just plug what you would plug in for mask
right into the brackets to filter effectively.
"""
print(mask.sum())
"""
This tells us how many of the passegers are children by summing up all of the rows that 
have a boolean value of one.
"""

# Plotting Basics:
# We almost always use matplotlib.pyplot to do simple plots of data
import matplotlib.pyplot as plt

plt.scatter(df['Age'], df['Fare'])
"""
This will create a scatter plot with the age as the x-axis and fare as the y-axis.
"""
plt.xlabel('Age')
plt.ylabel('Fare')
"""
This acually puts labels on the axes
"""

plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
"""
The class that the riders have is defined as the parameter. The riders can be in first,
second, or third class, which are denoted by the three colors in the plot.
"""
plt.plot([0, 80], [85, 5])
"""
This will plot a best fit line using data in the specified x-values (0 to 80) and
specified y-values (85 to 5).
"""

Classification:
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
"""
We set up the data so that it is completely numerical so that we can run a regression on it.
"""
import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
print(X)
print(y)
"""
Just checkin the work
"""

# Logistic Regression with SKLearn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
"""
Identify the model that we want to use
"""

X = df[['Fare', 'Age']].values
y = df['Survived'].values
model.fit(X, y)
"""
Setting up. Define our variables and input them into the model we want to use
"""

print(model.coef_, model.intercept_)
"""
Setup the best fit line
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
X = df[['Fare', 'Age']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)

print(model.coef_, model.intercept_)  # 0 = 0.0161594x + -0.01549065y + -0.51037152f
"""
Run the regression. The last line will give us the coefficients we can use for our best fit line.
This one is pretty simple so you probably would want to run it with more variables in the X Array.
"""

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)

model.predict(X)
"""
Setting up the model to predict given inputs for the variables in array X. Below gives a prediction as to
whether they survive when these conditions apply to that person
"""
print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))

print(model.predict(X[:5]))
"""
Above you predict the outcome of the first five rows using the prediction model. Below you print what the
results actually were
"""
print(y[:5])

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)
print((y == y_pred).sum())
print((y == y_pred).sum() / y.shape[0])
print(model.score(X, y))
"""
First you set up your prediction model, then in the second line you set up a boolean that counts all 
of the times that your prediction matched the reality. The third line calculates for what proportion of the
data does the predictor predict correctly.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
print(df.head())

X = df[cancer_data.feature_names].values
y = df['target'].values

model = LogisticRegression()
model.fit(X, y)
"""
model.fit(X,y) returned a convergence warning that the maximum number of iterations was reached so we
changed the solver to liblinear.
"""

model = LogisticRegression(solver='liblinear')
model.fit(X, y)

print("prediction for datapoint 0:", model.predict([X[0]]))
print(model.score(X, y))
"""
model.score tells you what proportion of the predictions were correct
"""

Model
Evaluation:
# Calculating Metrics in SKLearn:
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("accuracy:", accuracy_score(y, y_pred))  # Accuracy: What percent of the models predictions are correct
print("precision:",
      precision_score(y, y_pred))  # Precision: What percent of the models positive predictions are correct
print("recall:", recall_score(y, y_pred))  # Recall: What percent of the positive cases the model predicted correctly
print("f1 score:", f1_score(y, y_pred))  # F1 Score: The average of the precision and recall

# Confusion Matrix in SKLearn:
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y, y_pred))

# Training and Testing:

# Overfitting- When you create a model that performs better than it really would because it is performing on
# data that you know the outcome of.
# Test Set and Training Set- You normally want to take your data and use 70-80% of it as a training set
# to create your model and then use that model on the other 20-30% of the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
"""
First you have to import train_test_split from SKLearn and then you can split X and y into a train and test
case usnig the above line. Below just shows how the data is split.
"""
print("whole dataset:", X.shape, y.shape)
print("training set:", X_train.shape, y_train.shape)
print("test set:", X_test.shape, y_test.shape)

# Now that we know about training and test sets, we want to build model based on the training set data
# and evaluate the model based on how the model does on the test set.

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
y_pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
print("f1 score:", f1_score(y_test, y_pred))

# ROC Curve:

# Sensitivity vs. Specificity- Sensitivity is another word for recal which is
# P(true positive)/[p(true positive) + P(false negatives)]. Specificity is the true negative rate which can be calculated
# with the formula P(true negative)/[P(true negative) + P(false positive)]

# You can use SKLearn to calculate sensitivity and specificity of a prediction model for you. It should be
# noted that sensitivity and specificity can only be calculated when you know what the actual outcome is.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

from sklearn.metrics import recall_score

sensitivity_score = recall_score
print(sensitivity_score(y_test, y_pred))

from sklearn.metrics import precision_recall_fscore_support

print(precision_recall_fscore_support(y, y_pred))


def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]


print(specificity_score(y_test, y_pred))

print("sensitivity:", sensitivity_score(y_test, y_pred))
print("specificity:", specificity_score(y_test, y_pred))

# Adjusting the Logistic Regression Threshold in SKLearn
# SKLearn naturally finds the probability of an outcome and then rounds it to zero or one so you can see
# if the outcome is positive or negative. We can have SKLearn show us the probability it calculated instead of
# a zero or one.
model.predict_proba(X_test)
"""
Above will show us the probabilities of each person surviving or dying on the Titanic. Below we just filtered
for one column because once we know one column we know the other.
"""
model.predict_proba(X_test)[:, 1]

y_pred = model.predict_proba(X_test)[:, 1] > 0.75
"""
This threshold makes it harder for SKLearn to predict a positive outcome for surviving but we then can
feel more confident that the prediction is correct.
"""

print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))

# ROC Curve:
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()
"""
The closer the ROC curve is to the left corner, the better the model. Whatever specificity and sensitivity
correspond to the point closest to the left corner are the specifity and sensitivity that you should use
"""
X_train, X_test, y_train, y_test = train_test_split(X, y)

model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_proba1 = model1.predict_proba(X_test)
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba1[:, 1]))

model2 = LogisticRegression()
model2.fit(X_train[:, 0:2], y_train)
y_pred_proba2 = model2.predict_proba(X_test[:, 0:2])
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba2[:, 1]))
"""
AUC (Area Under Curve) is an empirical way to determine which model is better. After creating an ROC curve
you use the roc_auc_score function to see the area under each function. The greater the number the better.
This test lets us know how effective a logistic regression is for creating a predictive model for this data.
"""
# K-Fold Cross Validation:
# Multiple Training and Test Sets- When we have a small data set, the random assignment of data to a training
# set and a testing set can lead to differences in results if we only run the test once. To combat this, we
# can split the data into multiple chunks, lets say 5. You assign one chunk as a test set and the others as
# a training set. You create a model with this arrangement of test and training sets, and then switch the
# sets around until each of the chunks has been a test set, taking the measures of accuracy each time we
# create a new model and averaging the five results afterwards.

# K-Fold Cross Validation in SKLearn:
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
"""
Above you set up your k-fold cross validation by importing the data, setting your variables, and finally
priming the k-fold program with the necessary conditions. Below you define how your data will be split into 
train and test groups and print them to make sure they are right. Lastly, you want to define the train and test
groups for x and y so you can run the k-fold cross validation.
"""
kf = KFold(n_splits=5, shuffle=True)

splits = list(kf.split(X))
train_indices, test_indices = splits[0]
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)
print(np.mean(scores))
"""
Now we run the regression with five different splits of data, which is the k-fold cross validation.
It takes the five assortments of the data and finds the accuracy of each assortment, finally finding the mean
accuracy of these assortments to produce a more correct accuracy score. We now go back to using one model on
the whole data set instead of 5 different models, but the accuracy score is still the one we got when we ran
the K-fold cross validation.
"""

final_model = LogisticRegression()
final_model.fit(X, y)

# Model Comparison:
# You make a few different models and then use K-Fold Cross Validation for each model and compare them to see
# which is best

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
"""
Above you just import the programs you need and read in the data set. Below you set up your k-Fold cross
validation to be used on the models.
"""
kf = KFold(n_splits=5, shuffle=True)

X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values
"""
You have 3 different x data sets because you want to see which one is best at predicting. You only have
one y data set because you want to see which x data set is best at predicting the same y data set.
"""

print("Logistic Regression with all features")
score_model(X1, y, kf)
print()
print("Logistic Regression with Pclass, Sex & Age features")
score_model(X2, y, kf)
print()
print("Logistic Regression with Fare & Age features")
score_model(X3, y, kf)

Decision
Tree
Model:
# What is a Decision Tree?:
"""
So far we've only used logistic regression which is parametric, meaning that model it creates is based on 
defined parameters associated with coefficients. Decision trees are nonparametric so they won't be defined
by a set of parameters.
"""

# How to Build a Decision Tree:
"""
First you have to determine every possible split and then find which ones have the highest score. This
score is called the informational gain. This score is between 0 and 1 where one is a perfect split. You
ideally want to have all of one outcome on one side of the split and all of the other outcome on the other
outcome of of the split.
"""

# Gini Impurity:
"""
Measures how pure a data set is. A score of .5 means that the data is split exaclty 50-50
and a score of 0 or 1 means the data is all in the same class. A higher gini impurity score means that the
split you chose split the data more purely. GINI = 2(p)(1-p)
"""

# Entropy:
"""
Another measurement of the purity of the data set. It has a value between 0 and 1 where a 1
means that the data is completely impure (50-50 split) and a 0 means the data is pure (all the same class).
ENTROPY = -[plog2p + (1-p)log2(1-p)] <-- log base 2 of p or log base 2 of 1-p
"""

# Calculating Information Gain:
"""
Information gain uses one of the impurity scores from above to calculate how
effective the split that you chose was at splitting the data. A higher score is better. You have to do the
calculation for each possible split to see which one is best, but the computer can do the calculation for
you. For splits of numerical features, you have to try every possible threshold to see which one provides
the greates information gain.

INFORMATION GAIN = H(S) - (|A|/|S|)H(A) - (|B|/|S|)H(B)    S is the orginal dataset size, A and B are the
size of the datasets that you split dataset S into.
"""

# Decision Trees in SKLearn:
from sklearn.tree import DecisionTreeClassifier  # Import Statement

model = DecisionTreeClassifier()  # Define the model to be used
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
"""
Set our test and train sets and then a random state so we get different results every time
"""
model.fit(X_train, y_train)
print(model.predict([[3, True, 22, 1, 0, 7.25]]))  # Input the situation you want to have predicted
print("accuracy:", model.score(X_test, y_test))
y_pred = model.predict(X_test)
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))

# To increase the ability of the decision tree, you can run a K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True)
for criterion in ['gini', 'entropy']:
    print("Decision Tree - {}".format(criterion))
    accuracy = []
    precision = []
    recall = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
    print("accuracy:", np.mean(accuracy))
    print("precision:", np.mean(precision))
    print("recall:", np.mean(recall))

from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image

feature_names = ['Pclass', 'male']
X = df[feature_names].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)
"""
The tree diagram is saved as a file. To find the file just search teh file name followed by the format.
"""

# Overfitting of Tree Diagrams:
# Overfitting is when the model works well on the training set but not on the test set.

# Pruning:
"""
Setting rules to prevent the branches on the tree from getting too specific.
    Pre-Pruning: 3 techniques are used to limit the tree growth
        1) Max Depth: Limits the height of the tree. If you set it at 3, there will be at most 3 split for each
    data point
        2) Leaf Size: Don't split a node if the number of samples at that node is under a set threshold.
        3) Number of Leaf Nodes: Set a maximum number of possible leaf nodes in the tree
    You want to be careful with pruning because too much can make the model less useful. There is also no set
    way to go about pruning to get the best tree diagram. Trial and error is a guarantee.
"""

# Pruning with SKLearn:
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)
"""
You can put the parameters in the parentheses to change how the tree develops. You can use cross validation
to find the best values for the parameters.
"""

# Grid Search:
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]}
"""
We are giving the grid search parameters on which values to compare for each pruning method.
"""

gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
"""
You need to put in four parameters for the grid search: the model you are using (dt for data tree classifier),
the grid of values you are comparing (param_grid which we just defined), which metric to use to determine the
best model(default is accuracy but we are using f1), and lastly how many folds we want to use (5 in this case).
"""
gs.fit(X, y)
print("best params:", gs.best_params_)
print("best score:", gs.best_score_)
"""
The grid search finds the best values out of the ones we gave it to test and prints them for us. It should be
noted that these may not be the absolute best values.
"""

Random
Forest
Model:
# Random Forests with SKLearn:
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values
print('data dimensions', X.shape)
"""
Just some standard setup. Import general packages, set data frame, and define variables for model building.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
"""
Now we import the specific packages that we will use for the model and set up the train and test sets along
with defining the random state so we get unique results.
"""
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
"""
Define the model that will be used and fit it to the training set.
"""
first_row = X_test[0]
print("prediction:", rf.predict([first_row]))
print("true value:", y_test[0])
"""
Quick check to see if it is predicting and check if it's accurate.
"""

print("random forest accuracy:", rf.score(X_test, y_test))
"""
rf.score(X_test,y_test) takes the rf model that was created for the training set and tries it on the test
set. The number produced (0.9790209790209791) is the accuracy of the model (proportion of prediction that
are correct). This is above a decision tree model which only has an accuracy score of about 0.9.
"""

# Tuning a Random Forest Model:
"""
Random forest models are just made up of multiple decision trees, so the prepruning techniques for decision
trees (max depth, leaf size, number of leaf nodes) all can be used here too. Two new tuning techniques for
random forests are n_estimators which is the number of trees to combine to get a result, and max_features
which sets the maximum number of features to be considered at each split. Random forests don't usually
require as much tuning as decision trees.
"""
rf = RandomForestClassifier(max_features=5)
rf = RandomForestClassifier(n_estimators=15)

# Grid Search with Random Forests:
"""
For deicision trees, we gave a few values for each parameter of the tree for the grid search to compare
in order to find the values that will produce the best model. It is very similar with random forests, but
there are a few more parameters we can give inputs for.
"""

param_grid = {
    'n_estimators': [10, 25, 50, 75, 100],
}

gs = GridSearchCV(rf, param_grid, cv=5)
"""
In the decision tree model we used a different metric so we had to specifically change it but for this
random forest we are okay with using accuracy.
"""
gs.fit(X, y)
print("best params:", gs.best_params_)

# Elbow Graphs:
"""
Adding more trees to the model will never hurt performance but it will make the model more resource
intensive. Also, the improvement that comes with adding each tree will taper off. Thankfully, there are
elbow graphs to find the sweet spot where you get an effective model without creating a model that is too
complicated.
"""

n_estimators = list(range(1, 101))
param_grid = {
    'n_estimators': n_estimators,
}
rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)
gs.fit(X, y)
scores = gs.cv_results_['mean_test_score']

import matplotlib.pyplot as plt

scores = gs.cv_results_['mean_test_score']
plt.plot(n_estimators, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.show()
"""
If you run the graph you will see that the improvement in accuracy starts to taper off at about ten
trees. There is some shifting the graph as you increase the number of trees after 10 but that is probably
just due to chance.
"""

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)
"""
Armed with the knowledge that 10 is the optimal number of trees, we can now fit the random forest model
to X and y with ten trees and feel good about the quality of the model.
"""

# Feature Importances:
"""
Some features aren't as important to use when building a model and SKLearn has a package that can show
us which features are the most imporant. Only using certain features is helpful because it makes the model
less complicated and easier to interpret and explain to others. It can also improve the accuracy of the
model as well.
"""

rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X_train, y_train)

ft_imp = pd.Series(rf.feature_importances_, index=cancer_data.feature_names).sort_values(ascending=False)
ft_imp.head(10)
"""
We set up our model and then sort the columns by feature importance in descending order, taking a look
at the ten most important columns. The columns with "worst" seem to be appearing more often as important
columns, so below I created a data frame that includes just the columns with "worst".
"""
worst_cols = [col for col in df.columns if 'worst' in col]
print(worst_cols)

X_worst = df[worst_cols]
X_train, X_test, y_train, y_test = train_test_split(X_worst, y, random_state=101)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
"""
Interestingly enough, the accuracy of the random forest including only the features with the word "worst"
is more accurate than the forest that included all of the features.
"""

# Random Forest Pros and Cons:
"""
Random forests require less tuning than most other predictive models and work on almost every dataset
because they are not trying to fit any line or curve to the data. They are not as interpretable as decision
trees because they take multiple smaller decision trees and merge them to make a decision, something that
can be difficult to explain in context. They are also a relatively slower than decision trees, but not
too slow.
"""

Neural
Networks:
# Neurons in Nerual Networks:
"""
A neural network is a collection of many nodes that each do small calculations that lead to a 
meaningful calculation overall. They have inputs of x and outputs of y.

Like logistic regression, each neuron uses the general formula w1x1 + w2x2 + b where w1 and w2 are referred
to as weights and b is referred to as the bias. The result of this equation is plugged into the activation
function as its input. A common activation function is the sigmoid function [sigmoid(x) = 1/(1+e^-x)]. 
This equation will produce a number between zero and one. 
"""

# Activation Functions:
"""
There are three commonly used activation functions: sigmoid(look above), tanh, and ReLU.

tanh(x):
    Similar to sigmoid(x) but instead ranges from -1 to 1. 
    tanh(x) = sinh(x)/cosh(x) = (e^x - e^-x)/(e^x + e^-x)

ReLU:
    Stands for rectified linear unit
    ReLU = {0 if x<=0
           {x if x>0
"""

# Neural Network:
"""
To create a neural network, you combine multiple neurons and use the outputs of some as the inputs of others.
One type of neural network is the feed forward multi-layer perceptron. The feed forward means that the
neurons only send signals in one direction. Multi-layer perceptron means that it will have an input layer
(one node for each input), an output layer (a node for each output) and any number of hidden layers in
between that can take multiple inputs from the layers before them and produce an output that can go to 
multiple other nodes.

A single layer perceptron is a neural network without any hidden layers. These are incredibly rare.

One of the benefits of neural networks is that they can predict more than two values (0 or 1). It still only
produces one value at the end, but it has the ability to produce more than two. An example of this is a
program trying to predict what species an animal is. In the past, we would have had to settle for something
like cat or not cat, but now we can input a bunch of variables, it will apply those variables to the
neural network, and then it will choose from however many species we chose to define.
"""

# Training a Neural Network:
"""
When trainging a neural network, first you must define a loss function which measure how far from perfect
the network is. We train the neural network in an effort to minimize the loss function.

Backpropagation: The neural network can work backwards from the output node iteratively improving the
loss function each time. Eventually the loss function will be minimized and the optimal function will have
been found.
"""

# Neural Networks in SKLearn:
"""
Sometimes to test the neural network it can be helpful to create an artificial data set. When creating
a data set, 5 things have to be set: n_samples(number of datapoints), n_features(number of features),
n_informative(number of infomative features), n_redundant(number of redundant features), random_state
"""
from sklearn.datasets import make_classification

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)

from matplotlib import pyplot as plt

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=100, edgecolors='k', marker='^')
plt.show()
"""
It can be helpful to look at data in a visual way. Remember that this data is random.
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
"""
When you run the last 8 lines of code you get a convergence warning. The code shouldn't run and try to 
optimize forever so there is a set number of iterations that runs but in this case it didn't reach the
optimal coefficients for the nodes. To remedy this, increase the maximum number of iterations using the line
of code below.
"""
mlp = MLPClassifier(max_iter=1000)

"""
You can change some of the parameters of a multi-layer perceptron network such as the number of hidden
layers and number of nodes in each layer. The default is a single hidden layer of 100 nodes. Change the
numbers below to change the parameters.
"""
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50))

"""
Sometimes alpha, the step of the iterations needs to be changed (how much the coefficients change each
iteration). Decreasing alpha will make it more likely to reach the optimal solution, but you may also have
to increase the number of iterations. You may also want to change the solver, the algorithm that is used
to find the optimal solution. The three options are lbfgs, sgd, and adam. One may find the optimal solution
faster than another but it is just trial and error.
"""
# Predicting Handwritten Digits Example:
from sklearn.datasets import load_digits

X, y = load_digits(n_class=2, return_X_y=True)
print(X.shape, y.shape)
print(X[0])
print(y[0])
print(X[0].reshape(8, 8))

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(n_class=2, return_X_y=True)
plt.matshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())  # removes x tick marks
plt.yticks(())  # removes y tick marks
plt.show()
"""
The data set is how dark each pixel of an 8x8 picture is. by plotting the data set, we can see the number
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
"""
Above we set up our train and test sets, set up the classifier, and set it up to the data. Below we set x
to the first data point in our test set, shape it into an 8x8 array and plot it to see that it is zero.
the multi-layer perceptron neural network also predicted that the shape of the plot would be zero.
pretty crazy stuff
"""

x = X_test[0]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))

x = X_test[1]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))
"""
Correct again
"""

print(mlp.score(X_test, y_test))
"""
From the score we can see that the neural network had an accuracy of 100%
"""

"""
Now we are going to try it with all ten digits instead of zero and one
"""
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)

print(mlp.score(X_test, y_test))
"""
The MLP neural network is "only" 96% accurate. Below we figure out where it is getting tripped up.
"""

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

j = 0
print(incorrect[j].reshape(8, 8).astype(int))
print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])

"""
We can see that the first value that the neural network got wrong was when it predicted a 9 when in
actuality it was a 4. Understandable.
"""

# Visualizing MLP Weights:
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X5 = X[y <= '3']
y5 = y[y <= '3']
mlp = MLPClassifier(hidden_layer_sizes=(6,), max_iter=200, alpha=1e-4, solver='sgd', random_state=2)
mlp.fit(X5, y5)
"""
Importing a package, defining variables and the model, and fitting the model to the defined variables.
"""
print(mlp.coefs_)  # These are the weights for the nodes in the neural network
print(len(mlp.coefs_))  # We can see that the list of coefficients has two elements which makes sense; one
# is for the hidden layer and one is for the output layer

fig, axes = plt.subplots(2, 3, figsize=(5, 4))  # You can make multiple subplots instead of just oner
for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()

# Pros and Cons of Neural Networks:;
"""
Neural networks are super complicated and it is impossible to get into the nitty gritty and figure out
the network on its most fundamental levels. Because of this, it can feel kind of like putting inputs into 
a black box and getting outputs that you have to trust. They can also take a pretty long time to train
and refine. With all of that said, their performance is extraordinary. They open up unique opportunities by
easily tackling what would be a difficult problem for other methods and mostly tune themselves.
"""
