# Symbolic-Regression-GP

## Introduction
As "genetic programming for symbolic regression", a modification of the genetic algorithm package gplearn in Python, used for factor mining.

## Modules
### Function
Functions that calculate factors are implemented using the functional class `Function`, which includes 23 basic functions and 37 time series functions. All functions are essentially scalar functions, but because vectorized computation is used, both inputs and outputs are in vector form.

### Fitness
Fitness evaluation indicators are implemented using the functional class `Fitness`, which includes several fitness functions, mainly the Sharpe Ratio ("sharpe_ratio").

### Backtester
The vectorized factor backtesting framework follows the logic of first using the defined strategy function to turn the received "factor" into a "signal", and then using the signal processing function to turn the signal into an "asset" to implement backtesting. These two steps are combined in the functional class `Backtester`.

### SyntaxTree
The formula tree is used to write the calculation formula of the factor in prefix notation, and is represented using the formula tree `SyntaxTree`. Each formula tree represents a factor, and is composed of `Node`'s; each `Node` contains its own data, parent node, and child nodes. The `Node`'s own data can be a `Function`, variable, constant, or time-series constant.

The formula tree can be crossed over subtree mutated, hoisted, point mutated or reproduced (logic can be referred to gplearn).

### SymbolicRegressor
It contains the symbolic regression class (`SymbolicRegressor`), uses genetic algorithms to solve the symbolic regression problem, and defines some parameters during the genetic process, such as population size and number of generations.

## Usage

### Test
Like the example in `gplearn`, performing symbolic regression on $y=X_0^2 - X_1^2 + X_1 - 1$ with respect to $X_0$ and $X_1$ can yield the correct answer at around the 9th generation.

```Python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import *
from model.SymbolicRegressor import SymbolicRegressor



x0 = np.arange(-1, 1, 1 / 10.0)
x1 = np.arange(-1, 1, 1 / 10.0)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

ax = plt.figure().gca(projection="3d")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1, color="green", alpha=0.5)
plt.show()


rng = check_random_state(0)

# training samples
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1
X_train = pd.DataFrame(X_train, columns=["X0", "X1"])
y_train = pd.Series(y_train)

# testing samples
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1


sr = SymbolicRegressor(
    population_size=2000,
    tournament_size=20,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutate=0.1,
    p_hoist_mutate=0.1,
    p_point_mutate=0.05,
    init_depth=(6, 8),
    init_method="half and half",
    function_set=["add", "sub", "mul", "div", "square"],
    variable_set=["X0", "X1"],
    const_range=(0, 1),
    ts_const_range=(0, 1),
    build_preference=[0.75, 0.75],
    metric="mean absolute error",
    parsimony_coefficient=0.01,
)

sr.fit(X_train, y_train)


print(sr.best_estimator)
