# EvoStencils
Generating optimal iterative solvers through evolutionary computation
## Requirements
Python >= 3.6, sbt (for ExaStencils), MPI
## Setup
1. Clone and build the ExaStencils framework
```
git clone https://i10git.cs.fau.de/exastencils/release exastencils
cd exastencils
sbt compile
sbt assembly
```
2. Install required Python packages
```
pip install deap sympy mpi4py
```
3. Install LFA Lab (optional)
  Follow these instructions: https://hrittich.github.io/lfa-lab/install.html 
4. Set up the environment
```
source ./setup.sh
```
## Running
An example script for running an optimization can be found in the examples folder.

## Publications
    Schmitt J., Kuckuk S., Köstler H.:
    Optimizing Geometric Multigrid Methods with Evolutionary Computation
    Preprint available on arXiv: https://arxiv.org/abs/1910.02749
