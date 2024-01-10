from evostencils.ir import base, partitioning as part, system, transformations
from evostencils.ir import krylov_subspace
from evostencils.grammar import multigrid
from evostencils.code_generation import parser
import os
import subprocess
import math
import sympy
import time
from typing import List
import shutil

class InterGridOperations(Enum):
    Restriction = -1
    Interpolation = 1
    AltSmoothing = 0
class CorrectionTypes(Enum):
    Smoothing = 1
    CoarseGridCorrection = 0
class Smoothers(Enum):
    CGS_GE = 9
    Jacobi = 0
    GS_Forward = 13
    GS_Backward = 14
    NoSmoothing = -1

class ProgramGenerator:
    def __init__(self,min_level, max_level, mpi_rank=0,
                 solver_iteration_limit=None):
        self.min_level = min_level
        self.max_level = max_level
        self.mpi_rank = mpi_rank


# Affaire à suivre

# all includes in hyteg/apps/MultigridSTudies/multigridSTudies.cpp:

#include <cmath>

#include "core/Environment.h"
#include "core/Hostname.h"
#include "core/config/Config.h"
#include "core/math/Constants.h"
#include "core/timing/TimingJSON.h"

#include "hyteg/BuildInfo.hpp"
#include "hyteg/Git.hpp"
#include "hyteg/LikwidWrapper.hpp"
#include "hyteg/composites/P1StokesFunction.hpp"
#include "hyteg/composites/P1P1StokesOperator.hpp"
#include "hyteg/composites/P2P1TaylorHoodFunction.hpp"
#include "hyteg/composites/P2P1TaylorHoodStokesOperator.hpp"
#include "hyteg/composites/P2P2StokesFunction.hpp"
#include "hyteg/composites/P2P2UnstableStokesOperator.hpp"
#include "hyteg/dataexport/TimingOutput.hpp"
#include "hyteg/dataexport/VTKOutput/VTKOutput.hpp"
#include "hyteg/gridtransferoperators/P1P1StokesToP1P1StokesProlongation.hpp"
#include "hyteg/gridtransferoperators/P1P1StokesToP1P1StokesRestriction.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearProlongation.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearRestriction.hpp"
#include "hyteg/gridtransferoperators/P1toP1QuadraticProlongation.hpp"
#include "hyteg/gridtransferoperators/P1toP2Conversion.hpp"
#include "hyteg/gridtransferoperators/P2P1StokesToP2P1StokesProlongation.hpp"
#include "hyteg/gridtransferoperators/P2P1StokesToP2P1StokesRestriction.hpp"
#include "hyteg/gridtransferoperators/P2toP1Conversion.hpp"
#include "hyteg/gridtransferoperators/P2toP2QuadraticProlongation.hpp"
#include "hyteg/gridtransferoperators/P2toP2QuadraticRestriction.hpp"
#include "hyteg/memory/MemoryAllocation.hpp"
#include "hyteg/mesh/MeshInfo.hpp"
#include "hyteg/p1functionspace/P1ConstantOperator.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p2functionspace/P2ConstantOperator.hpp"
#include "hyteg/p2functionspace/P2Function.hpp"
#include "hyteg/petsc/PETScBlockPreconditionedStokesSolver.hpp"
#include "hyteg/petsc/PETScLUSolver.hpp"
#include "hyteg/petsc/PETScManager.hpp"
#include "hyteg/petsc/PETScVersion.hpp"
#include "hyteg/petsc/PETScWrapper.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/primitivestorage/Visualization.hpp"
#include "hyteg/solvers/CGSolver.hpp"
#include "hyteg/solvers/FullMultigridSolver.hpp"
#include "hyteg/solvers/GeometricMultigridSolver.hpp"
#include "hyteg/solvers/FlexibleMultigridSolver.hpp"
#include "hyteg/solvers/MinresSolver.hpp"
#include "hyteg/solvers/SORSmoother.hpp"
#include "hyteg/solvers/SymmetricSORSmoother.hpp"
#include "hyteg/solvers/UzawaSmoother.hpp"
#include "hyteg/solvers/controlflow/AgglomerationWrapper.hpp"
#include "hyteg/solvers/controlflow/TimedSolver.hpp"
#include "hyteg/solvers/preconditioners/stokes/StokesPressureBlockPreconditioner.hpp"
#include "hyteg/solvers/preconditioners/stokes/StokesVelocityBlockBlockDiagonalPreconditioner.hpp"

#include "sqlite/SQLite.h"
