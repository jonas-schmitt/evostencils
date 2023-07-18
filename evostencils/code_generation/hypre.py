import subprocess,re
from statistics import mean

from enum import Enum
class InterGridOperations(Enum):
    Restriction = -1
    Interpolation = 1
    AltSmoothing = 0

class Smoothers(Enum):
    Gauss_Elimination=9
    Jacobi = 0
    GS_Forward = 13
    GS_Backward = 14
    NoSmoothing = -1

class ProgramGenerator:
    def __init__(self,min_level, max_level) -> None:
        
        # INPUT
        self.min_level = min_level
        self.max_level = max_level

        # HYPRE FILES
        self.build_path = "../hypre/src/evo_test"
        self.problem = "ij"
        self.c_file_path= self.build_path + f"/{self.problem}.c"

        # TEMP OBJECTS
        self.list_states = []
        self.cycle_objs = []

        # AMG PARAMETERS
        self.intergrid_ops = [] # sequence of inter-grid operations in the multigrid solver -> describes the cycle structure. 
        self.smoothers = [] # sequence of different smoothers used across the AMG cycle.
        self.num_sweeps = [] # number of sweeps for each smoother.
        self.relaxation_weights = [] # sequence of relaxation factors for each smoother. 
        self.cgc_weights = [] # sequence of relaxations weights at intergrid transfer steps (meant for correction steps, weights in restriction steps is typically set to 1)

        #OUTPUT
        self.amgcycle= "" # the C expressions for the AMG specification i n hypre.

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
        self.amgcycle = ""

    def traverse_graph(self, expression): 
        expr_type = type(expression).__name__
        cur_lvl = expression.grid[0].level
        list_states = []
        cur_state = {'level':cur_lvl,'correction_type':None, 'component':None,'relaxation_factor':None}
        if expr_type == "Cycle" and expression not in self.cycle_objs:
            self.cycle_objs.append(expression)
            list_states = self.traverse_graph(expression.approximation) + self.traverse_graph(expression.correction)
            correction_expr_type = type(expression.correction.operand1).__name__
            if correction_expr_type  == "Prolongation":
                cur_state['correction_type']=0
                cur_state['component'] = -1
            elif correction_expr_type == "Inverse" :
                smoothing_operator = expression.correction.operand1.operand
                cur_state['correction_type']=1
                cur_state['component'] = smoothing_operator.smoother_type
            cur_state['relaxation_factor']=expression.relaxation_factor
            list_states.append(cur_state)
            return list_states
        elif expr_type == "Multiplication":
            list_states = self.traverse_graph(expression.operand2)
            op_type = type(expression.operand1).__name__
            if op_type == "CoarseGridSolver":
                cur_state['correction_type'] = 1
                cur_state['component'] = Smoothers.Gauss_Elimination
                cur_state['relaxation_factor'] = 1
                list_states.append(cur_state)
            return list_states
        elif "Residual" in expr_type:
            list_states = self.traverse_graph(expression.approximation) + self.traverse_graph(expression.rhs)
            return list_states
        else:
            return list_states
    def set_amginputs(self):
        cur_lvl = self.max_level
        for state in self.list_states:
            state_lvl = state['level']
            while cur_lvl > state_lvl: # restrict and go down the grid hierarchy until state_lvl is reached.
                self.intergrid_ops.append(InterGridOperations.Restriction)
                self.cgc_weights.append(1)
                if len(self.intergrid_ops)>len(self.smoothers): # no smoothing for the finer node.
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)
                cur_lvl -=1
            if state['correction_type']==1: # smoothing correction
                if state['component'] is Smoothers.Gauss_Elimination and state_lvl > 0:
                    if len(self.smoothers) < len(self.intergrid_ops):
                        self.smoothers.append(Smoothers.NoSmoothing)
                        self.relaxation_weights.append(0)
                        self.num_sweeps.append(1)
                    cycle_up = False
                    cur_lvl = state_lvl
                    while cur_lvl<=state_lvl:
                        if cur_lvl == 0:
                            self.smoothers.append(Smoothers.Gauss_Elimination)
                            self.num_sweeps.append(1)
                            self.relaxation_weights.append(1)
                            self.intergrid_ops.append(InterGridOperations.Interpolation)
                            self.cgc_weights.append(1)
                            cycle_up = True
                            cur_lvl+=1
                        elif not cycle_up:
                            self.smoothers.append(Smoothers.GS_Forward)
                            self.num_sweeps.append(1)
                            self.relaxation_weights.append(1)
                            self.intergrid_ops.append(InterGridOperations.Restriction)
                            self.cgc_weights.append(1)
                            cur_lvl-=1
                        elif cycle_up:
                            self.smoothers.append(Smoothers.GS_Backward)
                            self.num_sweeps.append(1)
                            self.relaxation_weights.append(1)
                            self.intergrid_ops.append(InterGridOperations.Interpolation)
                            self.cgc_weights.append(1)
                            cur_lvl+=1
                else:
                    self.smoothers.append(state['component'])
                    self.relaxation_weights.append(state['relaxation_factor'])
                    self.num_sweeps.append(1)
                if len(self.intergrid_ops) + 1 < len(self.smoothers):
                    self.intergrid_ops.append(InterGridOperations.AltSmoothing)
                    self.cgc_weights.append(0)
            elif state['correction_type'] == 0: # coarse grid correction
                if len(self.intergrid_ops) + 1 >len(self.smoothers): # no smoothing for the coarser node.
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)
                self.intergrid_ops.append(InterGridOperations.Interpolation)
                self.cgc_weights.append(state['relaxation_factor'])
            cur_lvl=state_lvl
        if len(self.smoothers)<len(self.intergrid_ops)+1:
            self.smoothers.append(Smoothers.NoSmoothing)
            self.num_sweeps.append(0)
            self.relaxation_weights.append(0)

    def generate_code(self):

        # 1. Read contents from the input/template file, identify where to print.
        # i. Specify the comment substrings
        comment_start = "USER_INPUTS"
        comment_end = "END_USER_INPUTS"

        # ii. Read the contents of the C file
        with open(self.c_file_path, 'r') as file:
            c_file_contents = file.readlines()

        # iii. Find the line numbers where the comments appear
        start_line = next(i for i, line in enumerate(c_file_contents) if comment_start in line)
        end_line = next(i for i, line in enumerate(c_file_contents) if comment_end in line)


        # 2. generate c expressions of the amg cycle. 
        # i. cycle structure
        self.amgcycle += f"cycle_num_nodes={len(self.smoothers)};\n"
        self.amgcycle +="HYPRE_Int icycling[]={"
        for ops in self.intergrid_ops:
            self.amgcycle+=f"{ops.value},"
        self.amgcycle=self.amgcycle[:-1] # remove the last comma.
        self.amgcycle+="};\n"
        # ii. relaxation types
        self.amgcycle += "HYPRE_Int irelaxtypes[]={"
        for smoother in self.smoothers:
            self.amgcycle+=f"{smoother.value},"
        self.amgcycle=self.amgcycle[:-1] # remove the last comma.
        self.amgcycle+="};\n"
        # iii. number of smoothing sweeps
        self.amgcycle += "HYPRE_Int inumsweeps[]={"
        for num_sweep in self.num_sweeps:
            self.amgcycle+=f"{num_sweep},"
        self.amgcycle=self.amgcycle[:-1] # remove the last comma.
        self.amgcycle+="};\n"
        # iv. relaxation weights
        self.amgcycle += "HYPRE_Real irelaxwts[]={"
        for relax_wts in self.relaxation_weights:
            self.amgcycle+=f"{relax_wts},"
        self.amgcycle=self.amgcycle[:-1] # remove the last comma.
        self.amgcycle+="};\n"
        # v. cgc weights
        self.amgcycle += "HYPRE_Real iedgewts[]={"
        for cgc_wts in self.cgc_weights:
            self.amgcycle+=f"{cgc_wts},"
        self.amgcycle=self.amgcycle[:-1] # remove the last comma.
        self.amgcycle+="};\n"

        # 3. append and print to file.
        # i. Modify the contents to replace the existing lines with the Python-generated C code snippet
        c_file_contents = c_file_contents[:start_line+1] + self.amgcycle.split('/n') + c_file_contents[end_line:]

        # ii. Write the modified contents back to the C file
        with open(self.c_file_path, 'w') as file:
            file.writelines(c_file_contents)

    def compile_code(self):
        subprocess.run(['make','clean'],cwd=self.build_path)
        subprocess.run(['make',self.problem],cwd=self.build_path)
    def execute_code(self):
        output = subprocess.run([self.build_path + "/" + self.problem], capture_output=True, text=True)
        # parse the output to extract wall clock time, number of iterations, convergence factor. 
        output_lines = output.stdout.split('\n')
        run_time = 1e10
        n_iterations = 1e10
        convergence_factor = 1e10
        solve_phase = True
        for line in output_lines:
            if "Solve phase" in line:
                solve_phase=True
            if "Convergence Factor" in line:
                match = re.search(r'\d+\.\d+', line)
                if match:
                    convergence_factor = float(match.group())
            elif "wall clock time" in line and solve_phase:
                match = re.search(r'\d+\.\d+', line)
                if match:
                    run_time = float(match.group())
            elif "Iterations" in line:
                match = re.search(r'\d+', line)
                if match:
                    n_iterations = int(match.group())
        
        if convergence_factor > 1:
            n_iterations = 1e10
        return run_time, convergence_factor, n_iterations
    
    def generate_and_evaluate(self, *args, **kwargs):
        expression = None
        time_solution_list = []
        convergence_factor_list = []
        n_iterations_list = []
        evaluation_samples = 1
        for arg in args:
            if type(arg).__name__ == 'Cycle':
                expression = arg

        if 'evaluation_samples' in kwargs:
            evaluation_samples = kwargs['evaluation_samples']
        
        self.reset()
        self.list_states = self.traverse_graph(expression)
   
        # fill in the AMG parameter list based on the sequence of AMG states visited in the GP tree.
        self.set_amginputs()
        
        # generate c code to set amg inputs
        self.generate_code()

        # compile and run
        self.compile_code()
        for _ in range(evaluation_samples):
            run_time, convergence, n_iterations = self.execute_code()
            time_solution_list.append(run_time)
            convergence_factor_list.append(convergence)
            n_iterations_list.append(n_iterations)

        return mean(time_solution_list), mean(convergence_factor_list), mean(n_iterations_list)
    
    # dummy functions to maintain compatibility in the optimisation pipeline
    def generate_storage(self, *args):
        empty_list = []
        return empty_list

    def initialize_code_generation(self, *args):
        pass

