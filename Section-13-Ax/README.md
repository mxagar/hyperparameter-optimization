# Hyperparameter Optimization with Ax: Adaptive Experimentation Platform

[Adaptive Experimentation Platform](https://ax.dev/): Ax is an accessible, general-purpose platform for understanding, managing, deploying, and automating adaptive experiments.

> Adaptive experimentation is the machine-learning guided process of iteratively exploring a (possibly infinite) parameter space in order to identify optimal configurations in a resource-efficient manner. Ax currently supports Bayesian optimization and bandit optimization as exploration strategies. Bayesian optimization in Ax is powered by [BoTorch](https://github.com/pytorch/botorch), a modern library for Bayesian optimization research built on PyTorch.

Features:

- Developed by Meta/Facebook.
- MIT license.
- Open Source: [Github](https://github.com/facebook/Ax)
- Modular: Easy to plug in new algorithms and use the library across different domains.
- Supports A/B Tests: Field experiments require a range of considerations beyond standard optimization problems.
- Production-Ready: Support for industry-grade experimentation and optimization management, including MySQL storage.

For installation:

```bash
conda install pytorch torchvision -c pytorch  # OSX only (details below)
pip install ax-platform

# Optional
pip install jupyter
pip install SQLAlchemy
```

## Tutorials

The package has a series of tutorials: [Ax Tutorials](https://ax.dev/tutorials/).

I downloaded and completed the following jupyter notebooks of the examples (in order):

- [`01_gpei_hartmann_loop.ipynb`](01_gpei_hartmann_loop.ipynb): Loop API.
- [`02_gpei_hartmann_service.ipynb`](02_gpei_hartmann_service.ipynb): Service API.
- [`03_gpei_hartmann_developer.ipynb`](03_gpei_hartmann_developer.ipynb): Developer API.
- [`04_visualizations.ipynb`](04_visualizations.ipynb): Visualization tools.
- [`05_generation_strategy.ipynb`](05_generation_strategy.ipynb): `GenerationStrategy`, optimization algorithm specification.
- [`06_scheduler.ipynb`](06_scheduler.ipynb): `Scheduler`.
- [`07_modular_botax.ipynb`](07_modular_botax.ipynb): interface to `BoTorch` module.
- [`08_tune_cnn.ipynb`](08_tune_cnn.ipynb): **(Bayesian) hyperparameter optimization for Pytorch**.

In the following, I comment the examples.

Note that the package has 3 APIs:

- Loop:
- Service:
- Developer:

The first 3 notebooks analyze those APIs with the standard[Hartmann 6](https://www.sfu.ca/~ssurjano/hart6.html) optimization problem. As described in part by ChatGPT

> The Hartmann 6 function is a well-known benchmark function often used in optimization and machine learning. It is a multi-dimensional function with six variables, defined as follows:
> ```
> f(x) = -sum(Ci * exp(-sum(Aij * (xi - Pij)^2)))
> ```
> where `x` is a vector of six variables, `Aij`, `Pij`, and `Ci` are constants specific to the function.
>
> The Hartmann 6 function is a global optimization problem, and the goal is to find the set of variables `x in (0,1)` that minimizes the function `f(x)`. It is a challenging function to optimize due to its complex shape and multiple local optima.
>
> The Hartmann 6 function is often used to evaluate and compare the performance of optimization algorithms, as it provides a difficult and well-defined optimization problem. It helps assess the ability of an optimization algorithm to find the global minimum in a high-dimensional space.
>
> Here are the specific values for the constants:
>
> ```plaintext
> A = [[10, 3, 17, 3.5, 1.7, 8],
>      [0.05, 10, 17, 0.1, 8, 14],
>      [3, 3.5, 1.7, 10, 17, 8],
>      [17, 8, 0.05, 10, 0.1, 14]]
> 
> P = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
>      [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
>      [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
>      [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
> 
> C = [1, 1.2, 3, 3.2]
> ```
> 
> The solution to the optimization problem:
>
> ```
> f(x_min) = -3.32237
> x_min = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
> ```

![Hartmann 6 Optimization Problem](../assets/hart62.png)

### Loop API

Notebook: [`01_gpei_hartmann_loop.ipynb`](01_gpei_hartmann_loop.ipynb)

:construction:

### Service API

Notebook: [`02_gpei_hartmann_service.ipynb`](02_gpei_hartmann_service.ipynb).

:construction:

### Developer API

Notebook: [`03_gpei_hartmann_developer.ipynb`](03_gpei_hartmann_developer.ipynb)

:construction:

### Visualization Tools

Notebook: [`04_visualizations.ipynb`](04_visualizations.ipynb).

:construction:

### Generation Strategy

Notebook: [`05_generation_strategy.ipynb`](05_generation_strategy.ipynb)

:construction:

### Scheduler

Notebook: [`06_scheduler.ipynb`](06_scheduler.ipynb).

:construction:

### BoTorch Interface

Notebook: [`07_modular_botax.ipynb`](07_modular_botax.ipynb)

:construction:

### Bayesian Hyperparameter Optimization for Pytorch

Notebook: [`08_tune_cnn.ipynb`](08_tune_cnn.ipynb).

:construction:
