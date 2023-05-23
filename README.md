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
	- [Section 2: Hyperparameter Tuning: Overview](#section-2-hyperparameter-tuning-overview)
		- [Notebooks and Code](#notebooks-and-code)
	- [Section 3: Performance Metrics](#section-3-performance-metrics)
		- [Notebooks and Code](#notebooks-and-code-1)
	- [Section 4: Cross-Validation](#section-4-cross-validation)
	- [Section 5](#section-5)
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

### Notebooks and Code

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

### Notebooks and Code

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



## Section 5



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

