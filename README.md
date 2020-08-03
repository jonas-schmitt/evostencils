# EvoStencils
Constructing efficient multigrid solvers through evolutionary computation
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

    Jonas Schmitt, Sebastian Kuckuk, and Harald Köstler
    Constructing Efficient Multigrid Solvers with Genetic Programming
    In Proceedings of the 2020 Genetic and Evolutionary Computation Conference (GECCO ’20)   
    Association for Computing Machinery, New York, NY, USA, 1012–1020  
    DOI: https://doi.org/10.1145/3377930.3389811

