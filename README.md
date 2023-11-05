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


