# EvoStencils
Generating optimal iterative solvers through evolutionary computation
## Requirements
Python >= 3.6, sbt (for ExaStencils)
## Setup
1. Clone and build the ExaStencils framework
```
git clone https://i10git.cs.fau.de/exastencils/release exastencils
cd exastencils
sbt compile
sbt assembly
```
2. Install LFA Lab  
  Follow these instructions: https://hrittich.github.io/lfa-lab/install.html  
3. Install required Python packages
```
pip install deap sympy
```
3. Set up the environment  
```
source ./setup.sh
```
## Running
An example script can be found in the examples folder. The Poisson examples from layer 2 should work without any further adaption.

## Publications
    Schmitt J., Kuckuk S., Köstler H.:
    Optimizing Geometric Multigrid Methods with Evolutionary Computation
    Submitted for publication 
    Preprint available on arXiv: https://arxiv.org/abs/1910.02749
