# EvoStencils

EvoStencils is a library for the automated design of **Multigrid (MG)** methods with **Grammar-Guided Genetic Programming (G3P)**. By treating the task of designing an efficient numerical solver as a program synthesis task, EvoStencils can discover MG methods of unprecedented algorithmic structure. To implement G3P, EvoStencils relies on [DEAP](https://github.com/DEAP/deap), while it utilizes the [ExaStencils](https://www.exastencils.fau.de) framework to automatically generate parallel C++ implementations based on an algorithmic representation of each MG-based solver specified in the form of a custom DSL. The following diagram provides an overview of EvoStencils' software architecture.

<img src="https://user-images.githubusercontent.com/5746840/212943618-23a53af2-59b9-4e55-8756-4dc5af49e5b5.svg" width="100%">

Results that were achieved using EvoStencils have been awarded with the [19th Annual Humies Gold Award](https://www.human-competitive.org/awards) for Human-Competitive Results.

EvoStencils is currently developed and maintained by [Jonas Schmitt](https://github.com/jonas-schmitt).

Examples of use can be found in [`notebooks`](https://github.com/jonas-schmitt/evostencils/tree/master/notebooks).

## Installation

EvoStencils requires a working installation of the message-passing interface `MPI` and depends on the Python packages `numpy`, `sympy`, `deap`, `mpi4py`. To install EvoStencils together with all required Python packages, invoke the following command:

```
pip install -e .
```

EvoStencils uses the [ExaStencils](https://www.exastencils.fau.de/) framework as a code generation backend, which requires a working installation of `java` (at least version 11) and `g++`. Under Linux (or WSL) these packages can be installed with the package manager. To clone and build ExaStencils in the current directory, you can use the following commands. 

```
wget -nc https://github.com/lssfau/ExaStencils/archive/refs/tags/v1.1.zip
wget -nc https://github.com/sbt/sbt/releases/download/v1.8.0/sbt-1.8.0.zip
unzip -n v1.1.zip && mv -vn ExaStencils-1.1 exastencils
unzip -n sbt-1.8.0.zip
cd exastencils
../sbt/bin/sbt compile
../sbt/bin/sbt assembly
cd ..
```

## Getting Started

- Look at the [tutorial](https://github.com/jonas-schmitt/evostencils/blob/master/notebooks/tutorial.ipynb).
- The folder [example_problems](https://github.com/jonas-schmitt/evostencils/blob/master/example_problems) contains a number of examples implemented in ExaStencils' DSL, which can be used as a drop in replacement for the two-dimensional Poisson problem covered in this tutorial.
- If you want to run EvoStencils directly, you can use the [optimize](https://github.com/jonas-schmitt/evostencils/blob/master/scripts/optimize.py) script as a starting point.
- Read our [journal paper](https://link.springer.com/10.1007/s10710-021-09412-w), which provides a good overview of EvoStencils' theoretical foundations.

## Citing

If you use or refer to EvoStencils in your work, please consider including the following citations:
<pre>
@misc{evostencils1,
      title={Automating the Design of Multigrid Methods with Evolutionary Program Synthesis}, 
      author={Jonas Schmitt},
      year={2023},
      eprint={2312.14875},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
</pre>

<pre>
@InProceedings{evostencils2,
  location = {Boston Massachusetts},
  title = {Evolving generalizable multigrid-based helmholtz preconditioners with grammar-guided genetic programming},
  url = {https://dl.acm.org/doi/10.1145/3512290.3528688},
  doi = {10.1145/3512290.3528688},
  eventtitle = {{GECCO} '22: Genetic and Evolutionary Computation Conference},
  pages = {1009--1018},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  publisher = {{ACM}},
  author = {Schmitt, Jonas and Köstler, Harald}
}
</pre>

<pre>
@Article{evostencils3,
  title = {{EvoStencils}: a grammar-based genetic programming approach for constructing efficient geometric multigrid methods},
  volume = {22},
  issn = {1389-2576, 1573-7632},
  url = {https://link.springer.com/10.1007/s10710-021-09412-w},
  doi = {10.1007/s10710-021-09412-w},
  shorttitle = {{EvoStencils}},
  pages = {511--537},
  number = {4},
  journaltitle = {Genetic Programming and Evolvable Machines},
  shortjournal = {Genet Program Evolvable Mach},
  author = {Schmitt, Jonas and Kuckuk, Sebastian and Köstler, Harald},
}
</pre>

## What is G3P and how is it related to Multigrid?

Grammar-Guided Genetic Programming (G3P) is a class of metaheuristic algorithms that construct programs based on the principle of natural evolution. G3P represents each program as a derivation tree, whose structure adheres to the rules of a formal grammar. To utilize G3P for the automated design of MG, EvoStencils formulates the rules of constructing a MG method in the form of a context-free grammar. Each derivation tree that results from the application of these rules thus represents a unique sequence of MG operations that operates on the given hierarchy of discretizations.

<img src="https://user-images.githubusercontent.com/5746840/212946312-8d8ff9a5-2999-4f20-8914-ac679bd6ce9e.svg" width="100%">

By imposing local changes on a certain derivation tree, G3P can alter the corresponding method's individual algorithmic steps without affecting its global structure, which offers the potential of discovering novel and potentially superior sequences of MG operations.

<img src="https://user-images.githubusercontent.com/5746840/212941118-971352f4-67b2-47b0-9cf3-ad3870dd7f1e.svg" width="100%">

