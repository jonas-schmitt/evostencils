# EvoStencils
Evolving efficient and generalizable multigrid methods with genetic programming
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
```
@inproceedings{evostencils2020,
  author = {Schmitt, Jonas and Kuckuk, Sebastian and K\"{o}stler, Harald},
  title = {Constructing Efficient Multigrid Solvers with Genetic Programming},
  year = {2020},
  isbn = {9781450371285},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3377930.3389811},
  doi = {10.1145/3377930.3389811},
  booktitle = {Proceedings of the 2020 Genetic and Evolutionary Computation Conference},
  pages = {1012–1020},
  numpages = {9},
  keywords = {geometric multigrid, context-free grammar, genetic programming, local fourier analysis, code generation},
  location = {Canc\'{u}n, Mexico},
  series = {GECCO '20}
}
```

