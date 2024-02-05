import subprocess,re
from statistics import mean
import os
import shutil
import numpy as np
import os

from enum import Enum
class InterGridOperations(Enum):
    Restriction = -1
    Interpolation = 1
    AltSmoothing = 0
class CorrectionTypes(Enum):
    Smoothing = 1
    CoarseGridCorrection = 0
class Smoothers(Enum):
    SOR = 7
    WeightedJacobi = 12
    SymmtericSOR = 10 # implemented only for 3D
    Chebyshev = 2
    Uzawa = 11
    GaussSeidel = 3
    SymmetricGaussSeidel = 9
    CGS_GE = 15
    NoSmoothing = 0

class ProgramGenerator:
    def __init__(self,min_level, max_level, mpi_rank=0,cgs_level=0) -> None:
        
        # INPUT
        self.min_level = min_level
        self.cgs_level = cgs_level
        self.max_level = max_level
        self.mpi_rank = mpi_rank

        # HYPRE FILES
        self.template_path = f"{os.getcwd()}/hyteg-build/apps/MultigridStudies"
        self.problem = "MultigridStudies"
        # generate build path 
        self.build_path = f"{self.template_path}_{self.mpi_rank}/"
        os.makedirs(self.build_path,exist_ok=True)
        # i. Get a list of all files in the template directory
        files = os.listdir(self.template_path) 
        files = [file for file in files if os.path.isfile(os.path.join(self.template_path, file))]
        for file in files:
            source_path = os.path.join(self.template_path,file)
            destination_path = os.path.join(self.build_path,file)
            shutil.copy(source_path,destination_path) # copy from source to destination
        # TEMP OBJECTS
        self.list_states = []
        self.cycle_objs = []
        self.n_individuals = 0

        # MG PARAMETERS
        self.intergrid_ops = [] # sequence of inter-grid operations in the multigrid solver -> describes the cycle structure. 
        self.smoothers = [] # sequence of different smoothers used across the MG cycle.
        self.num_sweeps = [] # number of sweeps for each smoother.
        self.relaxation_weights = [] # sequence of relaxation factors for each smoother. 
        self.cgc_weights = [] # sequence of relaxations weights at intergrid transfer steps (meant for correction steps, weights in restriction steps is typically set to 1)
        self.cgs_tolerance = None

        #OUTPUT
        self.mgcycle= "" # the command line arguments for MG specification in hyteg.

    @property
    def uses_FAS(self):
        return False

    def reset(self):
        self.list_states.clear()
        self.cycle_objs.clear()
        self.intergrid_ops.clear()
        self.smoothers.clear()
        self.num_sweeps.clear()
        self.relaxation_weights.clear()
        self.cgc_weights.clear()
        self.mgcycle = []

    def traverse_graph(self, expression): 
        expr_type = type(expression).__name__
        cur_lvl = expression.grid[0].level
        list_states = []
        cur_state = {'level':cur_lvl,'correction_type':None, 'component':None,'relaxation_factor':None,'additional_info':None}
        if expr_type == "Cycle" and expression not in self.cycle_objs:
            self.cycle_objs.append(expression)
            list_states = self.traverse_graph(expression.approximation) + self.traverse_graph(expression.correction)
            correction_expr_type = type(expression.correction.operand1).__name__
            if correction_expr_type  == "Prolongation":
                cur_state['correction_type']= CorrectionTypes.CoarseGridCorrection
                cur_state['component'] = -1
            elif correction_expr_type == "Inverse" :
                smoothing_operator = expression.correction.operand1.operand
                cur_state['correction_type']= CorrectionTypes.Smoothing
                cur_state['component'] = smoothing_operator.smoother_type
            cur_state['relaxation_factor']=expression.relaxation_factor
            list_states.append(cur_state)
            return list_states
        elif expr_type == "Multiplication":
            list_states = self.traverse_graph(expression.operand2)
            op_type = type(expression.operand1).__name__
            if op_type == "CoarseGridSolver":
                cur_state['correction_type'] = CorrectionTypes.Smoothing
                cur_state['component'] = Smoothers.CGS_GE
                cur_state['relaxation_factor'] = 1
                cur_state['additional_info'] = expression.operand1.additional_info
                list_states.append(cur_state)
            return list_states
        elif "Residual" in expr_type:
            list_states = self.traverse_graph(expression.approximation) + self.traverse_graph(expression.rhs)
            return list_states
        else:
            return list_states
        
    def set_mginputs(self):
        cur_lvl = self.max_level # finest level
        first_state_lvl = self.list_states[0]['level']
        # restrict from the finest level until first_state_lvl is reached
        n_cgs = 0
        while cur_lvl > first_state_lvl:
            self.smoothers.append(Smoothers.NoSmoothing)
            self.relaxation_weights.append(0)
            self.num_sweeps.append(0)
            self.intergrid_ops.append(InterGridOperations.Restriction)
            self.cgc_weights.append(1)
            cur_lvl -=1
        # loop through list_states
        for index,state in enumerate(self.list_states):
            state_lvl = state['level']
            assert state_lvl >= cur_lvl
            if state['correction_type']==CorrectionTypes.Smoothing: # smoothing correction
                if state['component'] == Smoothers.CGS_GE:
                    if state['additional_info']:
                        self.cgs_level = state['additional_info']['CGSlvl']
                        self.cgs_tolerance = state['additional_info']['CGStol']
                    while cur_lvl > self.cgs_level:
                        self.smoothers.append(Smoothers.GaussSeidel)
                        self.num_sweeps.append(1)
                        self.relaxation_weights.append(1)
                        self.intergrid_ops.append(InterGridOperations.Restriction)
                        self.cgc_weights.append(1)
                        cur_lvl -=1
                    self.smoothers.append(Smoothers.CGS_GE)
                    self.num_sweeps.append(1)
                    self.relaxation_weights.append(1)
                    while cur_lvl < state_lvl:
                        self.relaxation_weights.append(1)
                        self.intergrid_ops.append(InterGridOperations.Interpolation)
                        self.cgc_weights.append(1)
                        self.smoothers.append(Smoothers.GaussSeidel)
                        self.num_sweeps.append(1)
                        cur_lvl +=1
                else:
                    self.smoothers.append(state['component'])
                    self.relaxation_weights.append(state['relaxation_factor'])
                    self.num_sweeps.append(1)
            elif state['correction_type']==CorrectionTypes.CoarseGridCorrection: # coarse grid correction
                self.intergrid_ops.append(InterGridOperations.Interpolation)
                self.cgc_weights.append(state['relaxation_factor'])
            cur_lvl = state_lvl
            if index+1 < len(self.list_states):
                next_state_lvl = self.list_states[index+1]['level']
                next_state_correction_type = self.list_states[index+1]['correction_type']
                if next_state_lvl < cur_lvl:
                    if state['correction_type']==CorrectionTypes.CoarseGridCorrection:
                        self.smoothers.append(Smoothers.NoSmoothing)
                        self.num_sweeps.append(0)
                        self.relaxation_weights.append(0)
                    while cur_lvl > next_state_lvl: # restrict and go down the grid hierarchy until next_state_lvl is reached.
                        self.intergrid_ops.append(InterGridOperations.Restriction)
                        self.cgc_weights.append(1)
                        self.smoothers.append(Smoothers.NoSmoothing)
                        self.num_sweeps.append(0)
                        self.relaxation_weights.append(0)
                        cur_lvl -=1
                    self.smoothers.pop()
                    self.num_sweeps.pop()
                    self.relaxation_weights.pop()
                # if consecutive coarse grid corrections are performed 
                elif next_state_lvl > cur_lvl and state['correction_type']==next_state_correction_type==CorrectionTypes.CoarseGridCorrection:
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)
                # if consecutive smoothing steps are performed at the same level. 
                elif next_state_lvl == cur_lvl and state['correction_type']==next_state_correction_type==CorrectionTypes.Smoothing:
                    self.intergrid_ops.append(InterGridOperations.AltSmoothing)
                    self.cgc_weights.append(0) 
            elif index == len(self.list_states)-1:
                if state['correction_type']==CorrectionTypes.CoarseGridCorrection:
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)
 
    def generate_cmdline_args(self):
        # assert checks
        # sum of elements in intergrid_ops is zero, converting the enum to int
        assert sum([i.value for i in self.intergrid_ops]) == 0, "The sum of intergrid operations should be zero"
        # the grid hierarchy should be for self.max_level levels.
        assert min([sum([i.value for i in self.intergrid_ops[:j+1]]) for j in range(len(self.intergrid_ops))]) + self.max_level -self.cgs_level ==0, "The grid hierarchy should be for self.max_level - self.cgs_levels"
        # length of intergrid_ops is one less than length of smoothers
        assert len(self.intergrid_ops) == len(self.smoothers) - 1, "The number of intergrid operations should be one less than the number of nodes in the mg cycle"
        # length of smoothing weights is equal to length of smoothers and num_sweeps
        assert len(self.smoothers) == len(self.relaxation_weights) == len(self.num_sweeps), "The number of smoothing weights should be equal to the number of nodes in the mg cycle"
        # length of cgc weights is equal to length of intergrid_ops
        assert len(self.intergrid_ops) == len(self.cgc_weights), "The number of coarse grid correction weights should be equal to the number of intergrid operations in the mg cycle"
        # list to comma separated string
        def list_to_string(list):
            string = ""
            for item in list:
                # check if item is an enum
                if type(item).__name__ == 'Smoothers' or type(item).__name__ == 'InterGridOperations':
                    string += str(item.value) + ","
                else:
                    string += str(item) + ","
            return string[:-1]
        
        # generate the MG cycle string
        self.mgcycle = []
        self.mgcycle.append("-cycleStructure")
        self.mgcycle.append(list_to_string(self.intergrid_ops))
        self.mgcycle.append("-smootherTypes")
        self.mgcycle.append(list_to_string(self.smoothers))
        self.mgcycle.append("-smootherWeights")
        self.mgcycle.append(list_to_string(self.relaxation_weights))
        if not self.cgs_tolerance is None:
            self.mgcycle.append("-coarseGridResidualTolerance")
            self.mgcycle.append(str(self.cgs_tolerance))
            self.mgcycle.append("-minLevel")   
            self.mgcycle.append(str(self.cgs_level))


    def execute_code(self, cmd_args=[]):
        # run the code and pass the command line arguments from the input list
        output = subprocess.run([f"./{self.problem}"] + cmd_args, capture_output=True, text=True, cwd=self.build_path)
        # check if the code ran successfully
        if output.returncode != 0:
            print("error")
            print(output.args)
        # parse the output to extract wall clock time, number of iterations, convergence factor. 
        output_lines = output.stdout.split('\n')
        run_time = [1e100] * self.n_individuals
        n_iterations =[1e100] * self.n_individuals
        convergence_factor = [1e100] *  self.n_individuals
        i = 0 
        for line in output_lines:
            if "Convergence Factor" in line:
                match = re.search(r'Convergence Factor:\s*(\d+\.\d+)', line)
                if match:
                    convergence_factor[i] = float(match.group(1))
            elif "Solve Time" in line:
                match = re.search(r'Solve Time:\s*(\d+\.\d+)', line)
                if match:
                    run_time[i]=float(match.group(1))*1000 # convert to milliseconds
            elif "Iterations" in line:
                match = re.search(r'Number of Iterations:\s*(\d+)', line)
                if match:
                    n_iterations[i] = int(match.group(1))
        
        # if convergence factor is greater than 1, set n_iterations to 1e100

        n_iterations = [1e100 if cf > 1 else ni for cf, ni in zip(convergence_factor, n_iterations)]
        return run_time, convergence_factor, n_iterations
    def generate_and_evaluate(self, *args, **kwargs):
        expression_list = []
        time_solution_list = []
        convergence_factor_list = []
        n_iterations_list = []
        cmdline_args = [f"MultigridStudies.prm"]
        evaluation_samples = 1
        for arg in args:
            # get expression list from the input arguments
            if type(arg).__name__ == 'list':
                for cycle in arg:
                    if type(cycle).__name__ == 'Cycle':
                        expression_list.append(cycle)
            elif type(arg).__name__ == 'Cycle':
                expression_list.append(arg)

        self.n_individuals = len(expression_list)
        if 'evaluation_samples' in kwargs:
            evaluation_samples = kwargs['evaluation_samples']
        
        for expression in expression_list:
            self.reset()
            self.list_states = self.traverse_graph(expression)
    
            # fill in the MG parameter list based on the sequence of MG states visited in the GP tree.
            self.set_mginputs()
            
            # generate cmd line arguments to set mg inputs
            self.generate_cmdline_args()
            cmdline_args += self.mgcycle

        # run the code and pass the command line arguments from the list
        for _ in range(evaluation_samples):
            run_time, convergence, n_iterations = self.execute_code(cmdline_args)
            time_solution_list.append(run_time)
            convergence_factor_list.append(convergence)
            n_iterations_list.append(n_iterations)

        array_mean_time = np.atleast_1d(np.mean(time_solution_list,axis=0))
        array_mean_convergence = np.atleast_1d(np.mean(convergence_factor_list,axis=0))
        array_mean_iterations = np.atleast_1d(np.mean(n_iterations_list,axis=0))
        
        assert (array_mean_time.shape == array_mean_convergence.shape == array_mean_iterations.shape), "The shape of the output arrays with solver metrics (runtime, convergence, n_iterations) should be the same"
        return array_mean_time, array_mean_convergence, array_mean_iterations
    
    def generate_cycle_function(self, *args):
        expression = None
        for arg in args:
            if type(arg).__name__ == 'Cycle':
                expression = arg

        self.reset()
        self.list_states = self.traverse_graph(expression)
        self.set_mginputs()
        self.generate_cmdline_args()
        return str(self.mgcycle)

    # dummy functions to maintain compatibility in the optimisation pipeline
    def generate_storage(self, *args):
        empty_list = []
        return empty_list

    def initialize_code_generation(self, *args):
        pass

