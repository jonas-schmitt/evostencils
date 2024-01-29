# Build instructions for HyTeG

Clone via:

```
$ git clone --recurse-submodules https://i10git.cs.fau.de/hyteg/hyteg.git
```
Switch branch:
```
$ git checkout parthasarathy/FlexibleMultigridSolver
```
Create a build directory and invoke cmake:
```
$ mkdir hyteg-build 
$ cd hyteg-build
$ cmake ../hyteg
```
Build multigrid benchmark setting: 
```
hyteg-build $ cd apps/MultigridStudies
hyteg-build/apps/MultigridStudies $ make
```
