import subprocess,re,traceback
from statistics import mean
import os
import shutil
import numpy as np
import random
from enum import Enum
class InterGridOperations(Enum):
    Restriction = -1
    Interpolation = 1
    AltSmoothing = 0
    Terminate = -2
class CorrectionTypes(Enum):
    Smoothing = 1
    CoarseGridCorrection = 0
class Smoothers(Enum):
    CGS_GE = 9
    Jacobi = 0
    GS_Forward = 3
    GS_Backward = 4
    GS_Sym = 6
    l1Jacobi = 18
    l1GS_Forward = 13
    l1GS_Backward = 14
    NoSmoothing = -1

class ProgramGenerator:
    def __init__(self,min_level, max_level, hostname, mpi_rank=0) -> None:
        
        # INPUT
        self.min_level = min_level
        self.max_level = max_level
        self.mpi_rank = mpi_rank

        # HYPRE FILES
        self.template_path = "./evo_test"
        self.problem = "ij"
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
        # if rank is even, pin to socket 0, else pin to socket 1
        if self.mpi_rank % 2 == 0:
            self.mpi_pinning_arg = ['-host',f'{hostname}','-genv','I_MPI_PIN_PROCESSOR_LIST=0,1,2,3,4,5,6,7']
        else:
            self.mpi_pinning_arg = ['-host',f'{hostname}','-genv','I_MPI_PIN_PROCESSOR_LIST=36,37,38,39,40,41,42,43']

        # AMG PARAMETERS
        self.intergrid_ops = [] # sequence of inter-grid operations in the multigrid solver -> describes the cycle structure. 
        self.smoothers = [] # sequence of different smoothers used across the AMG cycle.
        self.relax_order = [] # sequence of relaxation orders for each smoother.
        self.num_sweeps = [] # number of sweeps for each smoother.
        self.relaxation_weights = [] # sequence of inner relaxation factors for each smoother. 
        self.relaxation_weights_outer = [] # sequence of outer relaxation factors for each smoother.
        self.cgc_weights = [] # sequence of relaxations weights at intergrid transfer steps (meant for correction steps, weights in restriction steps is typically set to 1)
        self.nx = 100
        self.ny = 100
        self.nz = 100
        self.cx = 0.001
        self.cy = 1
        self.cz = 1

        #OUTPUT
        self.amgcycle= "" # string representation of the AMG cycle
        self.amgcycle_args = [] # list of -flexamg_* flags for the ij binary

    @property
    def uses_FAS(self):
        return False

    def reset(self):
        self.list_states.clear()
        self.cycle_objs.clear()
        self.intergrid_ops.clear()
        self.smoothers.clear()
        self.relax_order.clear()
        self.num_sweeps.clear()
        self.relaxation_weights.clear()
        self.relaxation_weights_outer.clear()
        self.cgc_weights.clear()
        self.amgcycle = ""
        self.amgcycle_args = []

    def traverse_graph(self, expression): 
        expr_type = type(expression).__name__
        cur_lvl = expression.grid[0].level
        list_states = []
        cur_state = {'level':cur_lvl,'correction_type':None, 'component':None,'relaxation_factor':None, 'relaxation_factor_outer':None, 'relax_order':None}
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
                cur_state['relax_order'] = smoothing_operator.relax_order
            cur_state['relaxation_factor']=expression.relaxation_factor
            cur_state['relaxation_factor_outer']=expression.relaxation_factor_outer
            list_states.append(cur_state)
            return list_states
        elif expr_type == "Multiplication":
            list_states = self.traverse_graph(expression.operand2)
            op_type = type(expression.operand1).__name__
            if op_type == "CoarseGridSolver":
                cur_state['correction_type'] = CorrectionTypes.Smoothing
                cur_state['component'] = Smoothers.CGS_GE
                cur_state['relaxation_factor'] = 1
                cur_state['relaxation_factor_outer'] = 1
                list_states.append(cur_state)
            return list_states
        elif "Residual" in expr_type:
            list_states = self.traverse_graph(expression.approximation) + self.traverse_graph(expression.rhs)
            return list_states
        else:
            return list_states
        
    def set_amginputs(self):
        cur_lvl = self.max_level # finest level
        first_state_lvl = self.list_states[0]['level']
        # restrict from the finest level until first_state_lvl is reached
        while cur_lvl > first_state_lvl:
            self.smoothers.append(Smoothers.NoSmoothing)
            self.relax_order.append(0)
            self.relaxation_weights.append(0)
            self.relaxation_weights_outer.append(0)
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
                    self.smoothers.append(Smoothers.CGS_GE)
                    self.relax_order.append(0)
                    self.num_sweeps.append(1)
                    self.relaxation_weights.append(1)
                    self.relaxation_weights_outer.append(1)
                else:
                    self.smoothers.append(state['component'])
                    self.relax_order.append(state['relax_order'])
                    self.relaxation_weights.append(state['relaxation_factor'])
                    self.relaxation_weights_outer.append(state['relaxation_factor_outer'])
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
                        self.relax_order.append(0)
                        self.num_sweeps.append(0)
                        self.relaxation_weights.append(0)
                        self.relaxation_weights_outer.append(0)
                    while cur_lvl > next_state_lvl: # restrict and go down the grid hierarchy until next_state_lvl is reached.
                        self.intergrid_ops.append(InterGridOperations.Restriction)
                        self.cgc_weights.append(1)
                        self.smoothers.append(Smoothers.NoSmoothing)
                        self.relax_order.append(0)
                        self.num_sweeps.append(0)
                        self.relaxation_weights.append(0)
                        self.relaxation_weights_outer.append(0)
                        cur_lvl -=1
                    self.smoothers.pop()
                    self.relax_order.pop()
                    self.num_sweeps.pop()
                    self.relaxation_weights.pop()
                    self.relaxation_weights_outer.pop()
                # if consecutive coarse grid corrections are performed 
                elif next_state_lvl > cur_lvl and state['correction_type']==next_state_correction_type==CorrectionTypes.CoarseGridCorrection:
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.relax_order.append(0)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)
                    self.relaxation_weights_outer.append(0)
                # if consecutive smoothing steps are performed at the same level. 
                elif next_state_lvl == cur_lvl and state['correction_type']==next_state_correction_type==CorrectionTypes.Smoothing:
                    self.intergrid_ops.append(InterGridOperations.AltSmoothing)
                    self.cgc_weights.append(0) 
            elif index == len(self.list_states)-1:
                if state['correction_type']==CorrectionTypes.CoarseGridCorrection:
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.relax_order.append(0)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)
                    self.relaxation_weights_outer.append(0)

        # add termination state at the end of the cycle
        self.intergrid_ops.append(InterGridOperations.Terminate)
        # add 0 to cgc_weights so that length of cgc_weights is equal to length of intergrid_ops
        self.cgc_weights.append(0)
 
    def generate_cmdline_args(self):
        # assert checks
        # sum of elements in intergrid_ops is zero, converting the enum to int
        assert sum([i.value for i in self.intergrid_ops]) == 0, "The sum of intergrid operations should be zero"
        # the grid hierarchy should be for self.max_level levels.
        assert min([sum([i.value for i in self.intergrid_ops[:j+1]]) for j in range(len(self.intergrid_ops))]) + self.max_level - self.min_level ==0, "The grid hierarchy should be for self.max_level levels"
        # length of intergrid_ops is one less than length of smoothers
        assert len(self.intergrid_ops) == len(self.smoothers), "The number of intergrid operations should be equal to the number of nodes in the amg cycle"
        # length of smoothing weights is equal to length of smoothers and num_sweeps
        assert len(self.smoothers) == len(self.relax_order) == len(self.relaxation_weights) == len(self.relaxation_weights_outer) == len(self.num_sweeps), "The number of smoothing weights should be equal to the number of nodes in the amg cycle"
        # length of cgc weights is equal to length of intergrid_ops
        assert len(self.intergrid_ops) == len(self.cgc_weights), "The number of coarse grid correction weights should be equal to the number of intergrid operations in the amg cycle"
        def to_str(lst):
            return ','.join(str(v.value if hasattr(v, 'value') else v) for v in lst)

        # cycle_struct: intergrid op after each smoother node, -2 as terminator on the last
        cycle_struct = [op.value for op in self.intergrid_ops] + [-2]
        cgc_scaling  = list(self.cgc_weights) + [0.0]

        self.amgcycle_args = [
            '-flexamg_cycle_struct', to_str(cycle_struct),
            '-flexamg_relax_types',  to_str(self.smoothers),
            '-flexamg_relax_orders', to_str(self.relax_order),
            '-flexamg_cgc_scaling',  to_str(cgc_scaling),
            '-flexamg_relax_weights', to_str(self.relaxation_weights),
            '-flexamg_outer_weights', to_str(self.relaxation_weights_outer),
        ]
        self.amgcycle = ' '.join(self.amgcycle_args)

    def compile_code(self):
        subprocess.run(['make','clean'],cwd=self.build_path)
        subprocess.run(['make',self.problem],cwd=self.build_path)
    def execute_code(self, cmd_args=[]):
        mpiarg = ["mpirun","-np","8"] + self.mpi_pinning_arg
        try:
            output = subprocess.run(mpiarg + [self.build_path + self.problem] + cmd_args, capture_output=True, text=True)
        except Exception:
            print("An error occurred:")
            traceback.print_exc()
            return 1e100, 1e100, 1e100
        output_lines = output.stdout.split('\n')
        run_time = 1e100
        n_iterations = 1e100
        convergence_factor = 1e100
        solve_phase = False
        for line in output_lines:
            if "Solve phase times" in line:
                solve_phase = True
            if "Convergence Factor" in line:
                match = re.search(r'\d+\.\d+', line)
                if match:
                    convergence_factor = float(match.group())
            elif "wall clock time" in line and solve_phase:
                match = re.search(r'\d+\.\d+', line)
                if match:
                    run_time = float(match.group()) * 1000
                solve_phase = False
            elif "Iterations" in line:
                match = re.search(r'\d+', line)
                if match:
                    n_iterations = int(match.group())
        if convergence_factor > 1:
            n_iterations = 1e100
        return run_time, convergence_factor, n_iterations
    def generate_and_evaluate(self, *args, **kwargs):
        expression_list = []
        rhs_newton_itr = 1
        base_cmdline_args = ["-P","4","2","1","-fromfile",f"/home/vault/iwia/iwia058h/8_procs_89100_dofs/ij_A_8procs04_02_01_004_00{rhs_newton_itr}","-rhsfromfile",f"/home/vault/iwia/iwia058h/8_procs_89100_dofs/ij_b_8procs04_02_01_004_00{rhs_newton_itr}","-pout","0","-solver","3","-th","0.8","-rlx_down","6","-rlx_up","6","-k","100","-mg_max_iter","500","-precon_cycles","1","-falgout","-mxrs","0.9","-tol","1e-4","-atol","1e-8"]
        for arg in args:
            if type(arg).__name__ == 'list':
                for cycle in arg:
                    if type(cycle).__name__ == 'Cycle':
                        expression_list.append(cycle)
            elif type(arg).__name__ == 'Cycle':
                expression_list.append(arg)
        evaluation_samples = kwargs.get('evaluation_samples', 1)

        time_results = []
        convergence_results = []
        iteration_results = []
        for expression in expression_list:
            self.reset()
            self.list_states = self.traverse_graph(expression)
            self.set_amginputs()
            self.generate_cmdline_args()
            ind_times, ind_cf, ind_ni = [], [], []
            for _ in range(evaluation_samples):
                rt, cf, ni = self.execute_code(base_cmdline_args + self.amgcycle_args)
                ind_times.append(rt)
                ind_cf.append(cf)
                ind_ni.append(ni)
            time_results.append(mean(ind_times))
            convergence_results.append(mean(ind_cf))
            iteration_results.append(mean(ind_ni))

        array_mean_time = np.atleast_1d(np.array(time_results))
        array_mean_convergence = np.atleast_1d(np.array(convergence_results))
        array_mean_iterations = np.atleast_1d(np.array(iteration_results))
        assert (array_mean_time.shape == array_mean_convergence.shape == array_mean_iterations.shape), "The shape of the output arrays with solver metrics (runtime, convergence, n_iterations) should be the same"
        return array_mean_time, array_mean_convergence, array_mean_iterations
    
    def generate_cycle_function(self, *args):
        expression = None
        for arg in args:
            if type(arg).__name__ == 'Cycle':
                expression = arg

        self.reset()
        self.list_states = self.traverse_graph(expression)
        self.set_amginputs()
        self.generate_cmdline_args()
        return self.amgcycle

    # dummy functions to maintain compatibility in the optimisation pipeline
    def generate_storage(self, *args):
        empty_list = []
        return empty_list

    def initialize_code_generation(self, *args):
        pass

