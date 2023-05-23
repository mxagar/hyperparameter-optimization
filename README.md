# Hyperparameter tuning for Machine Learning

![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/solegalli/hyperparameter-optimization/blob/master/LICENSE](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/solegalli/hyperparameter-optimization/blob/master/LICENSE)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/), modified by [Mikel Sagardia](https://mikelsagardia.io/)

Originally published by [Soledad Galli @ Train in Data](https://www.trainindata.com/) in May, 2021.

[<img src="./logo.png" width="248">](https://www.courses.trainindata.com/p/hyperparameter-optimization-for-machine-learning)

Modified by [Mikel Sagardia](https://mikelsagardia.io/) while folowing the associated Udemy course [Hyperparameter Optimization for Machine Learning
](https://www.udemy.com/course/hyperparameter-optimization-for-machine-learning).

## Overview of Topics

1. **Cross-Validation**
	1. K-fold, LOOCV, LPOCV, Stratified CV
	2. Group CV and variants
	3. CV for time series
	4. Nested CV

2. **Basic Search Algorithms**
	1. Manual Search, Grid Search and Random Search

3. **Bayesian Optimization**
	1. with Gaussian Processes
	2. with Random Forests (SMAC) and GBMs
	3. with Parzen windows (Tree-structured Parzen Estimators or TPE)

4. **Multi-fidelity Optimization**
	1. Successive Halving
	2. Hyperband
	3. BOHB

5. **Other Search Algorthms**
	1. Simulated Annealing
	2. Population Based Optimization

6. **Gentetic Algorithms**
	1. CMA-ES	

7. **Python tools**
	1. Scikit-learn
	2. Scikit-optimize
	3. Hyperopt
	4. Optuna
	5. Keras Tuner
	6. SMAC
	7. Others
	8. **Added by me: Ax**

## Table of Contents

- [Hyperparameter tuning for Machine Learning](#hyperparameter-tuning-for-machine-learning)
	- [Overview of Topics](#overview-of-topics)
	- [Table of Contents](#table-of-contents)
	- [Setup](#setup)
		- [Datasets](#datasets)
	- [Section 2: Hyperparameter Tuning: Overview](#section-2-hyperparameter-tuning-overview)
		- [Examples and Code](#examples-and-code)
	- [Section 3: Performance Metrics](#section-3-performance-metrics)
		- [Examples and Code](#examples-and-code-1)
	- [Section 4: Cross-Validation](#section-4-cross-validation)
		- [Cross-Validation Schemes](#cross-validation-schemes)
		- [Hyperparameter Tuning with Different Cross-Validation Schemes](#hyperparameter-tuning-with-different-cross-validation-schemes)
		- [Special Cross-Validation Schemes: Non-Independent Data](#special-cross-validation-schemes-non-independent-data)
		- [Nested Cross-Validation](#nested-cross-validation)
	- [Section 5: Basic Search Algorithms](#section-5-basic-search-algorithms)
		- [Manual Search](#manual-search)
		- [Grid Search](#grid-search)
		- [Random Search](#random-search)
		- [Random Search with Other Packages](#random-search-with-other-packages)
	- [Section 6](#section-6)
	- [Section 7](#section-7)
	- [Section 8](#section-8)
	- [Section 9](#section-9)
	- [Section 10](#section-10)
	- [Section 11](#section-11)
	- [Section 12](#section-12)
	- [Section 13: Ax Platform](#section-13-ax-platform)

## Setup

Links:

- [Online Course](https://www.courses.trainindata.com/p/hyperparameter-optimization-for-machine-learning)
- [Github repository](https://github.com/trainindata/hyperparameter-optimization)
- [Slides](https://www.dropbox.com/sh/wzbn528sxwdc22k/AAD7IJej-9NwcD5bHK8bbMDka?dl=0)
- [Datasets: MNIST Kaggle](https://www.kaggle.com/c/digit-recognizer/data): rename the `train.csv` file to be `mnist.csv`.

Environment:

```bash
conda create --name hyp pip python=3.8
conda activate hyp
python -m pip install -r requirements.txt
```

### Datasets

Apart from the [MNIST Kaggle](https://www.kaggle.com/c/digit-recognizer/data) dataset, the [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) dataset is used.

```python
from sklearn.datasets import load_breast_cancer

breast_cancer_X, breast_cancer_y = load_breast_cancer(return_X_y=True)
X = pd.DataFrame(breast_cancer_X)
y = pd.Series(breast_cancer_y).map({0:1, 1:0})
```

## Section 2: Hyperparameter Tuning: Overview

Hyperparameters are those parameters which are not learnt during taining, but chosen by the user. They can be used to:

- Improve convergence
- Improve performance
- Prevent overfitting
- etc.

However, the hyperparameters affect the parameters learnt by the model.

Typical hyperparameters for Random Forests and Gradient Boosted Trees:

- Number of trees
- The depth of the tree
- Learning rate (GBMs)
- The metric of split quality
- The number of features to evaluate at each node
- The minimum number of samples to split the data further

The most important hyperparameters are those that optimize the generalization error, which is not necessarily the loss.

Some hyperparameters do not affect the model performance; we need to try all possible value combinations, but that comes with an inrceased computational cost.

A hyperparameter search consists of:

- Hyperparameter space: the hyperparameters we are going to test and their possible values.
- A method for sampling candidate hyperparameters
  - Manual Search
  - Grid Search
  - Random Search
  - Bayesian Optimization
  - Other
- A cross-validation scheme
- A performance metric to minimize (or maximize)

Two important concepts that come up often when talking about hyperparameters optimization:

- **Hyperparameter response surface**: that's the value of the decision metric as function of the hyperparameter values; we want to minimize it. Notation: `lambda = argmin(Phi(lambda))`, i.e., `lambda` are the hyperparameter values and `Phi(lambda)` is the response surface, i.e., the decision metric.
- **Low effective dimension**: the repsonse surface is very sensitive to some hyperparameters and it doesn't change with others; we want to find which hyperparameters affect `Phi`.

### Examples and Code

We can evaluate these two concepts, e.g., when we use the `GridSearchCV` from Scikit-Learn. These notebooks show how to do that:

- [`02-01-Response-Surface.ipynb`](./Section-02-Hyperparamter-Overview/02-01-Response-Surface.ipynb)
- [`02-02-Low-Effective-Dimension.ipynb`](./Section-02-Hyperparamter-Overview/02-02-Low-Effective-Dimension.ipynb)

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# model
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=1, random_state=0, n_jobs=4)

# hyperparameter space
rf_param_grid = dict(
    n_estimators=[50, 100, 200]
    max_depth=[2, 3, 4],
)

# search
# the available metrics can be passed as a string
# https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
reg = GridSearchCV(rf_model,
				   rf_param_grid,
                   scoring='neg_mean_squared_error',
				   cv=3)
search = reg.fit(X, y)

# best hyperparameters
search.best_params_

# We have all the CV results for each parameter combination
# in this attribute!
search.cv_results_
# mean_fit_time
# std_fit_time
# mean_score_time
# std_score_time
# params
# param_max_depth
# param_n_estimators
# split0_test_score
# split1_test_score
# split2_test_score
# mean_test_score
# std_test_score
# rank_test_score

# Get the CV results
results = pd.DataFrame(search.cv_results_)[['params', 'mean_test_score']]

## 1. Plot response surface
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

# depth
x = [r['max_depth'] for r in results['params']]
# number of trees
y = [r['n_estimators'] for r in results['params']]
# performance
z = results['mean_test_score']

# plotting
ax.scatter(x, y, z,)
ax.set_title('Response Surface')
ax.set_xlabel('Tree depth')
ax.set_ylabel('Number of trees')
ax.set_zlabel('negative rmse')
plt.show()

## 2. Plot effect
results = pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
results.index = rf_param_grid['n_estimators']

results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylim(0.2, 0.6)
plt.ylabel('Mean r2 score')
plt.xlabel('Number of trees')

# We can also plot the peformance in the hyperparameter space
results.sort_values(by='mean_test_score', ascending=False, inplace=True)
results.reset_index(drop=True, inplace=True)

results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylim(-0.3, 0)
plt.ylabel('Mean False Negative Rate')
plt.xlabel('Hyperparameter space')
```

## Section 3: Performance Metrics

Classification metrics:

- Dependent on probability:
  - Accuracy: correct / total
    - Confusion matrix
  - Precision: Positive Predictive Value
  - Recall = Sensitivity: True positive rate
  - F1
  - False Positive Rate
  - False Negative Rate
- Independent from probability, aggregate values
  - ROC: Receiver-Operator Characteristic Curve
    - ROC-AUC: ROC area under the curve
- Loss: `-(y*log(p) + (1-y)*log(1-p))`

Regression metrics:

- Square Error
- Mean Square Error, MSE
- Root Mean Square Error, RMSE
- Mean Absolute Error, MAE
- **R2: how much of the total variance that exists in our date is explained by the model**

### Examples and Code

We want to minimize some metrics and maximize others. In Scikit-Learn, the metrics are always *maximized*; therefore, we use `neg_mae`, i.e., the `-MAE` is maximized:

[Scikit-Learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

If we want to use a metric which is not defined, i.e., a custom metric defined by us, we can define it with `make_scorer`. That makes a lot of sense in some cases because there is not an available meric or because we have very specific needs. For instance, in cancer prediction, we want to minimize the false negatives, so we need the *False Negative Rate*; additionally, we might want to set the probability threshold manually! Choosing the correct metric completely changes both the *response surface* and the *effect of the hyperparameters*!

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix

def fnr(y_true, y_pred):
    """False negative rate.
	This is essential for cases like cancer detection,
	in which a false negative has really bad consequences."""
	# I we need it, we can pass a probability and define our own threshold:
	# 	y_pred = np.where(y_pred > 0.37, 1, 0)
	# BUT we need to set needs_proba=True in make_scorer
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()  
    FNR = fn / (tp + fn)
    
    return FNR

fnr_score = make_scorer(
    fnr,
    greater_is_better=False, # smaller is better
    needs_proba=False, # specify y_pred needs to be a probability
)

rf = RandomForestClassifier(n_estimators=100,
							max_depth=1,
							random_state=0,
							n_jobs=4)

# hyperparameter space
params = dict(
    n_estimators=[10, 50, 200],
    max_depth=[1, 2, 3],
)

# search
# we can use our custom/defined score/metric
clf = GridSearchCV(rf,
                   params,
                   scoring=fnr_score,
                   cv=5)

search = clf.fit(X, y)

# best hyperparameters
search.best_params_
```

## Section 4: Cross-Validation

There is always a bias-variance tradeoff in our models:

- If the model has bias, it underfits the data, i.e., it is too simplistic and it does not capture essential behaviors.
- If the model has variance, it overfits the data, i.e., it is too complex and it captures noise.

We want a model which generalizes well. To achieve that, we need to evaluate our final model on a test split which has never been seen. When we are choosing between different hyperparameter combinations, we cannot use that test split either, because then we would be fitting the model to that split, i.e., we'd be leaking information from the test set to the training.

To perform a correct hyperparameter tuning, we have two major ways:

1. Use 3 splits: `train`, `validation`, `test`. We train the model with `train` for each hyperparameter combination and evaluate them with `validation`. Then, we pick the best model and evaluate it with `test`. If the model generalizes well, the best `validation` performance should be similar to the `test` performance. This approach requires a large dataset; keep in mind that small `train` sets lead to bad performing models (biased).
2. We apply cross-validation. We have 2 splits `train` and `test`, but `train` is further split in `k` non-overlapping subsets; then, for each hyperparameter combination, we perform `k` trainings (i.e., `k` models) in which `k-1` subsets are used and the remaining is employed for validation. At the end, we get the mean validation error and the `test` split is used for the final evaluation.

In gradient descent algorithms, the validation performance can decrease in the beginning but start increasing past a point, in which we start overfitting.

See: [Scikit-Learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

The main goal or strategy is:

- Split the dataset into `train` and `test`.
- Perform hyperparameter tuning with cross-validation using `train`.
- Obtain the final performance metrics (with std. dev./error) for `train` and `val` and check that they considerably overlap. The validation error/metric is the first estimation of the generalization error.
- Compute the performance metrics for the unseen `test`; this is the real **generalization error**.
- Verify that the performance of `test` is in the confidence interval of `train`.

### Cross-Validation Schemes

If the data is **independent and identically distributed (iid.)**, we can use these cross-validation schemes:

- K-fold: the scheme explained before, typically with `k in [5,10]`; the larger the `k`, the more risk of high variance. But try to use at least `k = 5`.
- Leave-one-out: `k = n`, i.e., we leave one data point out for validation. We have `n` models to train, very expensive; additionally, all models have a very similar `train` set, which can lead to high variance. It could be used for continuous metrics, but doesn't work well for classification metrics, in general.
- Leave-P-out: we leave P points out and carry out combinatorics of points; again, many models, much moe than only leaving one out.
- Repeated K-fold: repeat K-fold `n` times, each time with data points shuffled. A good alternative.
- Stratified K-fold: K-fold for classification in which class ratios are preserved in each folded split. This is important for imbalanced classification problems; try to use it in any classification problem.

```python
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    LeaveOneOut,
    LeavePOut,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

# Logistic Regression
logit = LogisticRegression(
    penalty ='l2', C=10, solver='liblinear', random_state=4, max_iter=10000)

# K-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=4)
# Repeated K-Fold Cross-Validation
cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=4,
)
# Leave One Out Cross-Validation
cv = LeaveOneOut()
# Leave P Out Cross-Validation
cv = LeavePOut(p=2)
# Leave P Out Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

# estimate generalization error
clf =  cross_validate(
    logit,
    X_train, 
    y_train,
    scoring='accuracy',
    return_train_score=True,
    cv=cv, # we could also pass an int, and it performs k-fold with n_splits
)

# the score in the subset that was left; k values
clf['test_score']

# the score of the subset used for training; k values
clf['train_score']

# The idea is that both distirbutions should overlap considerably
print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
```

### Hyperparameter Tuning with Different Cross-Validation Schemes

If we use Scikit-Learn, we need to use `RandomSearchCV` or `GridSearchCV` for hyperparameter tuning wth cross-validation. The code below is very similar to the previous one, but we pass our cross-validation scheme to `GridSearchCV` instead of `cross_validate()`.

```python
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    LeaveOneOut,
    LeavePOut,
    StratifiedKFold,
    GridSearchCV, # RandomSearchCV
    train_test_split,
)

# hyperparameter space
param_grid = dict(
    penalty=['l1', 'l2'],
    C=[0.1, 1, 10],
)

# K-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=4)
# Repeated K-Fold Cross-Validation
cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=4,
)
# Leave One Out Cross-Validation
cv = LeaveOneOut()
# Leave P Out Cross-Validation
cv = LeavePOut(p=2)
# Leave P Out Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

# search
grid_search =  GridSearchCV(
    logit,
    param_grid,
    scoring='accuracy',
    cv=cv, # any CV scheme, as defined in the previous section
    refit=True, # refits best model to entire dataset
)

search = grid_search.fit(X_train, y_train)

# best hyperparameters
search.best_params_

# get all results
results = pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
```

### Special Cross-Validation Schemes: Non-Independent Data

The introduced cross-validation schemes were for **independent and identically distirbuted (iid)** data; however, some datasets don't fulfill the necessary requirements for that:

- Grouped data: multiple observations from the same subject; repeated measurements.
- Time series
 
For **grouped data**, the goal here would be to measure whether the model generalizes well for other subjects! To achieve that, the data from each subject is treated as a group and groups are moved around together; thus, we can use *grouped* K-Fold, or leave-one/p-out. In other words, groups are treated as data points before.

For **time series**, we want to predict future values. Thus, we create groups of time periods (e.g., months) and move them together. The additional constraint is that they are not shuffled and that if we predict month `M`, we use only the prior months as the train set.

See: 

- [Cross-validation iterators for grouped data](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
- [Cross validation of time series data](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data)

Example from [`04-03-Group-Cross-Validation.ipynb`](./Section-04-Cross-Validation/04-03-Group-Cross-Validation.ipynb), where we simulate repeated measurements of different patients:

```python
# Logistic Regression
logit = LogisticRegression(
    penalty ='l2', C=10, solver='liblinear', random_state=4, max_iter=10000)

# Group K-Fold Cross-Validation
# We are going to define groups as patient numbers
cv = GroupKFold(n_splits=5)
# Similarly, we have:
cv = LeaveOneGroupOut()

# estimate generalization error
clf =  cross_validate(
    logit,
    X_train.drop('patient', axis=1), # drop the patient column, this is not a predictor
    y_train,
    scoring='accuracy',
    return_train_score=True,
    cv=cv.split(X_train.drop('patient', axis=1), y_train, groups=X_train['patient']),
)
```

### Nested Cross-Validation

This topic is relevant for competitions.

When we train different models with a hyperparameter search scheme (e.g., grid search) and a cross-validation scheme (e.g., K-fold), the generalization error is still positively biased. The reason is because in every `k` training, the validation subset is part of the training subest for another training; thus, we are leaking training information to the validation.

A solution to that consists in performing nested cross-validation: we perform a cross-validation within a cross-validation. This is an expensive approach, but used when we need a good generalization error estimation.

![Nested Cross-Validation](./assets/nested_cv.jpg)

See notebook: [`04-04-Nested-Cross-Validation.ipynb`](./Section-04-Cross-Validation/04-04-Nested-Cross-Validation.ipynb).

## Section 5: Basic Search Algorithms

Things to consider:

- Number of hyperparameters of the machine learning model
- The low effective dimension: select hyperperameters that have an effect, and regions or spaces which are associated with changes in the response surface.
- The nature of the parameters: discrete, continuous, categorical; each type leads to different space definition startegies.
- The computing resources available to us: in general, the more combinations we try, the better, but that's not always true nor feasible.

### Manual Search

We manually try with `cross_validate()` different hyperparameter values and obtain the `test` errors (mean and standard deviation).

The idea is to obtain ranges of values that lead to models that generalize well, i.e., the error/score distirbutions of the `train` and `test/val` splits in the cross validation overlap and the unseen `test` split is contained in them.

When we have the approximate values, we define the ranges/values in a search scheme, e.g., `GridSearchCV` or `RandomSearchCV`.

### Grid Search

Gris search is an exhaustive method: it tests all possible combinations of hyperparameter values we specify.

Grid search is very expensive, because the trials grow exponentially; however, it can be parallelized.

In practice, grid search is not enough, because it rarely finds the best sets of hyperparameters in the complete space. Instead, we use grid search for an initial search and then we fine tune with the results we obtain from it.

Important notebook, where all these concepts are implemented: [`02-Grid-Search.ipynb`](./Section-05-Basic-Search-Algorithms/02-Grid-Search.ipynb). Nothing really new is introduced, but these ideas are correctly implemented:

- A first broad grid search is done and results are colledted in a data frame.
- The effect of each parameter is analyzed.
- The search space is narrowed down and grid search is applied again.

```python
# set up the model
gbm = GradientBoostingClassifier(random_state=0)

# determine the hyperparameter space: 60 possible combinations
param_grid = dict(
    n_estimators=[10, 20, 50, 100],
    min_samples_split=[0.1, 0.3, 0.5],
    max_depth=[1,2,3,4,None],
    )

# set up the search
search = GridSearchCV(gbm, param_grid, scoring='roc_auc', cv=5, refit=True)

# find best hyperparameters
search.fit(X_train, y_train)

# get results
results = pd.DataFrame(search.cv_results_) # 60 x 16

# we can order the different models based on their performance
results.sort_values(by='mean_test_score', ascending=False, inplace=True)
results.reset_index(drop=True, inplace=True)
results[[
    'param_max_depth', 'param_min_samples_split', 'param_n_estimators',
    'mean_test_score', 'std_test_score',
]].head()

# plot model performance and error
results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)
plt.ylabel('Mean test score')
plt.xlabel('Hyperparameter combinations')

# get overall generalization error
X_train_preds = search.predict_proba(X_train)[:,1]
X_test_preds = search.predict_proba(X_test)[:,1]
print('Train roc_auc: ', roc_auc_score(y_train, X_train_preds))
print('Test roc_auc: ', roc_auc_score(y_test, X_test_preds))
# Train roc_auc:  1.0
# Test roc_auc:  0.996766607877719

# let's make a function to evaluate the model performance based on
# single hyperparameters
def summarize_by_param(hparam):
    tmp = pd.concat([
        results.groupby(hparam)['mean_test_score'].mean(),
        results.groupby(hparam)['mean_test_score'].std(),
    ], axis=1)
    tmp.columns = ['mean_test_score', 'std_test_score']
    
    return tmp

# check the effect of the parameter n_estimators
tmp = summarize_by_param('param_n_estimators')
tmp['mean_test_score'].plot(yerr=[tmp['std_test_score'], tmp['std_test_score']], subplots=True)
plt.ylabel('roc-auc')
# now repeat for all other parameters
# select the regions to refine in a new grid search

# determine the hyperparameter space
param_grid = dict(
    n_estimators=[60, 80, 100, 120],
    max_depth=[2,3],
    loss = ['deviance', 'exponential'],
    )

# set up the search
search = GridSearchCV(gbm, param_grid, scoring='roc_auc', cv=5, refit=True)

# find best hyperparameters
search.fit(X_train, y_train)

# the best hyperparameters are stored in an attribute
search.best_params_

# get all results
results = pd.DataFrame(search.cv_results_)
# perform comparisons/analysis if desired

# compute new generalization error, which should be better
X_train_preds = search.predict_proba(X_train)[:,1]
X_test_preds = search.predict_proba(X_test)[:,1]
print('Train roc_auc: ', roc_auc_score(y_train, X_train_preds))
print('Test roc_auc: ', roc_auc_score(y_test, X_test_preds))
# Train roc_auc:  0.9999999999999999
# Test roc_auc:  0.9973544973544973
```

Note that in some cases a parameter value in a model enables other parameters. To deal with that, we can simply define 2+ dictionaries in the parameter grid; for instance, `SVC`:

```python
# set up the model
svm = SVC(random_state=0)

# determine the hyperparameter space
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# set up the search
search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=3, refit=True)

# find best hyperparameters
search.fit(X_train, y_train)
```

### Random Search

While `GridSearchCV` explores all combinations, `RandomSearchCV` explores a randomly chosen subset from all possible combinations. It is very effective, because many parameters have regions with low effect. Therefore, if we have many hyperparameters, it is the preferred choice; if we have few hyperparameters, grid search is fine.

Random search is also well suited for continuous hyperparameters, because the search can draw values from a distribution; in contrast, grid search requires a manual definition of values to be tested. Therefore: **instead of entering single values to `RandomSearchCV`, we should pass distirbutions to maximize its power!**.

**Important note**: There is a probabilistic explanation that states that, independently of the search space, with 60 combinations we have a 95% probability of finding a set in neighborhood of the optimal set (top 5%). More on that: [The "Amazing Hidden Power" of Random Search?](https://stats.stackexchange.com/questions/561164/the-amazing-hidden-power-of-random-search).

For the rest, the usage of `GridSearchCV` and `RandomSearchCV` is quite similar.

```python
from scipy import stats

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)

# set up the model
gbm = GradientBoostingClassifier(random_state=0)

# determine the hyperparameter space
# NOTE: we use distributions!
# To get random values:
# - integers: stats.randint.rvs(1, 5)
# - continuous/float: stats.uniform.rvs(0, 1)
# Here we don't call rvs() because we're ot drawing the numbers
# but passing the distribution
param_grid = dict(
    n_estimators=stats.randint(10, 120),
    min_samples_split=stats.uniform(0, 1),
    max_depth=stats.randint(1, 5),
    loss=('deviance', 'exponential'),
    )

# set up the search
search = RandomizedSearchCV(gbm,
                            param_grid,
                            scoring='roc_auc',
                            cv=5,
                            n_iter = 60, # this is the number of combinations we want
                            random_state=10,
                            n_jobs=4,
                            refit=True)

# find best hyperparameters
search.fit(X_train, y_train)

# the best hyperparameters are stored in an attribute
search.best_params_

# we also find the data for all models evaluated
results = pd.DataFrame(search.cv_results_)

# we can order the different models based on their performance
results.sort_values(by='mean_test_score', ascending=False, inplace=True)
results.reset_index(drop=True, inplace=True)
results[[
    'param_max_depth', 'param_min_samples_split', 'param_n_estimators',
    'mean_test_score', 'std_test_score',
]].head()

# plot model performance and error
results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)
plt.ylabel('Mean test score')
plt.xlabel('Hyperparameter combinations')

# generalization error
X_train_preds = search.predict_proba(X_train)[:,1]
X_test_preds = search.predict_proba(X_test)[:,1]
print('Train roc_auc: ', roc_auc_score(y_train, X_train_preds))
print('Test roc_auc: ', roc_auc_score(y_test, X_test_preds))

# let's make a function to evaluate the model performance based on
# single hyperparameters
def summarize_by_param(hparam):
    tmp = pd.concat([
        results.groupby(hparam)['mean_test_score'].mean(),
        results.groupby(hparam)['mean_test_score'].std(),
    ], axis=1)
    tmp.columns = ['mean_test_score', 'std_test_score']
    
    return tmp

# performance change for n_estimators
tmp = summarize_by_param('param_n_estimators')
tmp['mean_test_score'].plot(yerr=[tmp['std_test_score'], tmp['std_test_score']], subplots=True)
plt.ylabel('roc-auc')
```

### Random Search with Other Packages

Usually, we'll perform random searches with Scikit-Learn, but it's possible to do it with other packages, too, such as:

- [Scikit-Optimize](https://scikit-optimize.github.io/stable/)
- [Hyperopt](http://hyperopt.github.io/hyperopt/)

These packages were introduces to run Bayesian optimization, but they support also random search.

Notebooks where it is shown how:

- [`05-Randomized-Search-with-Scikit-Optimize.ipynb`](./Section-05-Basic-Search-Algorithms/05-Randomized-Search-with-Scikit-Optimize.ipynb)
- [`06-Randomized-Search-with-Hyperopt.ipynb`](./Section-05-Basic-Search-Algorithms/06-Randomized-Search-with-Hyperopt.ipynb)




## Section 6



## Section 7



## Section 8



## Section 9



## Section 10



## Section 11



## Section 12



## Section 13: Ax Platform

This section was not in the original course and it was added by me; it introduces [Ax, the Adaptive Experimentation Platform](https://ax.dev/).

There is a custom `README.md` in the folder [Section-13-Ax/](./Section-13-Ax/), along with a custom file of requirements.

