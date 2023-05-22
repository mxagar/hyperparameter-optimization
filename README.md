![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/solegalli/hyperparameter-optimization/blob/master/LICENSE](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/solegalli/hyperparameter-optimization/blob/master/LICENSE)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)

## Hyperparameter tuning for Machine Learning - Code Repository

Published May, 2021

[<img src="./logo.png" width="248">](https://www.courses.trainindata.com/p/hyperparameter-optimization-for-machine-learning)

## Links

- [Online Course](https://www.courses.trainindata.com/p/hyperparameter-optimization-for-machine-learning)
- [Github repository](https://github.com/trainindata/hyperparameter-optimization)
- [Slides](https://www.dropbox.com/sh/wzbn528sxwdc22k/AAD7IJej-9NwcD5bHK8bbMDka?dl=0)
- [Datasets: MNIST Kaggle](https://www.kaggle.com/c/digit-recognizer/data): rename the `train.csv` file to be `mnist.csv`.

## Setup

```bash
conda create --name hyp pip python=3.8
conda activate hyp
python -m pip install -r requirements.txt
```

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

- [Hyperparameter tuning for Machine Learning - Code Repository](#hyperparameter-tuning-for-machine-learning---code-repository)
- [Links](#links)
- [Setup](#setup)
- [Overview of Topics](#overview-of-topics)
- [Table of Contents](#table-of-contents)
- [Section 2: Hyperparameter Tuning: Overview](#section-2-hyperparameter-tuning-overview)
- [Section 3: Performance Metrics](#section-3-performance-metrics)
- [Section 4](#section-4)
- [Section 5](#section-5)
- [Section 6](#section-6)
- [Section 7](#section-7)
- [Section 8](#section-8)
- [Section 9](#section-9)
- [Section 10](#section-10)
- [Section 11](#section-11)
- [Section 12](#section-12)
- [Section 13: Ax Platform](#section-13-ax-platform)

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

We can evaluate these two concepts, e.g., when we use the `GridSearchCV` from Scikit-Learn. These notebooks show how to do that:

- [`02-01-Response-Surface.ipynb`](./Section-02-Hyperparamter-Overview/02-01-Response-Surface.ipynb)
- [`02-02-Low-Effective-Dimension.ipynb`](./Section-02-Hyperparamter-Overview/02-02-Low-Effective-Dimension.ipynb)

## Section 3: Performance Metrics





## Section 4



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

