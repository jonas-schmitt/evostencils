# EvoStencils
EvoStencils is an open source tool for the grammar-based design of geometric multigrid methods for solving partial differential equations (PDEs) with stencil-based discretizations. 
With the use of grammar-guided genetic programming (GGGP) it can alter a multigrid cycle's components on each discretization level individually and is, therefore, capable of evolving algorithms whose construction has been previously impossible.

EvoStencils is purely written in [Python](https://www.python.org/) and utilizes the [DEAP](https://github.com/DEAP/deap) framework as genetic programming backend. To automatically generate efficient implementations of a multigrid solver, EvoStencils builds upon the [ExaStencils](https://www.exastencils.fau.de/) framework, which is capable of generating highly optimized and scalable solver code that utilizes the hardware resources of recent supercomputers.

EvoStencils has been developed at the Chair for System Simulation of Friedrich-Alexander University Erlangen-Nürnberg (FAU).
In case you run into any troubles using EvoStencils, please contact [jonas.schmitt@fau.de](https://www.cs10.tf.fau.de/person/jonas-schmitt/).


## Installation
#### Requirements: Python >= 3.6, sbt (for ExaStencils), MPI
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
4. Clone EvoStencils
```
git clone https://github.com/jonas-schmitt/evostencils
```
5. Set up the environment
```
cd evostencils
source ./setup.sh
```
## Running
An example script for running an optimization can be found in the scripts folder.
```
python scripts/optimize.py
```
## Publications
To refer to EvoStencils, please cite the following publications.
```
@article{evostencils2021,
  author={Schmitt, Jonas and Kuckuk, Sebastian and K{\"o}stler, Harald},
  title={EvoStencils: a grammar-based genetic programming approach for constructing efficient geometric multigrid methods},
  journal={Genetic Programming and Evolvable Machines},
  year={2021},
  month={Sep},
  day={03},
  issn={1573-7632},
  doi={10.1007/s10710-021-09412-w},
  url={https://doi.org/10.1007/s10710-021-09412-w}
}
```
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

