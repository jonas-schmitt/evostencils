# EvoStencils
Generating optimal iterative solvers through evolutionary computation
## Requirements
Python >= 3.6
## Setup
1. Install LFA Lab  
  Follow these instructions: https://hrittich.github.io/lfa-lab/install.html  
2. Install required Python packages
```
pip install deap sympy
```
3. Set up the environment  
```
source ./setup.sh
```
## Running
An example script can be found in the examples folder. Note that a working version of the ExaStencils framework (http://exastencils.org - currently not open source) is required for the code generation.

## Publications
    Schmitt J., Kuckuk S., Köstler H.:
    Optimizing Geometric Multigrid Methods with Evolutionary Computation
    Submitted for publication 
    Preprint available on arXiv: https://arxiv.org/abs/1910.02749
