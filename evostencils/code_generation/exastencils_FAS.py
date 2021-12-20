from evostencils.code_generation.layer4 import *
from evostencils.code_generation import parser
from evostencils.expressions import base
from statistics import mean
import subprocess
import math


class ProgramGeneratorFAS:
    def __init__(self, problem_name, solution, rhs, residual, FASApproximation, restriction, prolongation, op_linear, op_nonlinear, fct_name_mgcycle, fct_cgs, fct_smoother=None, mpi_rank=0):
        # ExaStencils configuration
        self.problem_name = problem_name
        self.platform_file = "../Compiler/mac.platform"
        self.build_path = "../example_problems/"
        self.exastencils_compiler = "../Compiler/Compiler.jar"
        self.layer3_file = f"{problem_name}/2D_FD_Poisson_fromL2.exa3"
        self.settings_file = f"{problem_name}/{problem_name}_froml4.settings"
        self.settings_file_generated = f"{problem_name}/{problem_name}_froml4_{mpi_rank}.settings"
        self.knowledge_file = f"{problem_name}/{problem_name}.knowledge"
        self.exa_file_template = f"{problem_name}/{problem_name}_template.exa4"
        self.exa_file_out = f"{problem_name}/{problem_name}_{mpi_rank}.exa4"
        self.min_level = 0
        self.max_level = 0
        self.dimension = 0
        self.mpi_rank = mpi_rank

        # get info from knowledge file and set dimensionality, max, min grid levels
        self.dimension, self.min_level, self.max_level = parser.extract_knowledge_information(self.build_path, self.knowledge_file)

        # get function names if already defined in exa4
        self.fct_CGS = fct_cgs
        self.fct_smoother = fct_smoother

        # bool to clip multiple paths to the same 'rhs' node i.e. smoothing, restriction
        self.update_rhs = {}

        # init fields/stencils for each level
        self.solution = {}
        self.rhs = {}
        self.residual = {}
        self.FASApproximation = {}
        self.restriction = {}
        self.prolongation = {}
        self.op_linear = {}
        if op_nonlinear is not None:
            self.op_nonlinear = {}
        else:
            self.op_nonlinear = None

        for i in range(self.min_level, self.max_level + 1):
            self.solution[i] = Field(solution, i)
            self.rhs[i] = Field(rhs, i)
            self.residual[i] = Field(residual, i)
            self.FASApproximation[i] = Field(FASApproximation, i)
            self.restriction[i] = Stencil(restriction, i)
            self.prolongation[i] = Stencil(prolongation, i)
            self.op_linear[i] = Stencil(op_linear, i)
            self.update_rhs[i] = True
            if op_nonlinear is not None:
                self.op_nonlinear[i] = Stencil(op_nonlinear, i)

        # mgcycle generated
        self.fct_name = fct_name_mgcycle + f"@{self.max_level}"
        self.fct_mgcycle = None
        self.fct_body = []

        # variables to maintain compatibility during optimisation pipeline
        self.equations, self.operators, self.fields = \
            parser.extract_l2_information(self.build_path + self.layer3_file, self.dimension, None)
        size = 2 ** self.max_level
        grid_size = tuple([size] * self.dimension)
        h = 1 / (2 ** self.max_level)
        step_size = tuple([h] * self.dimension)
        tmp = tuple([2] * self.dimension)
        self.coarsening_factor = [tmp for _ in range(len(self.fields))]
        self.finest_grid = [base.Grid(grid_size, step_size, self.max_level) for _ in range(len(self.fields))]

    def traverse_graph(self, expression):
        def level(obj):
            return obj.grid[0].level

        def updateSolution():
            solution = self.traverse_graph(expression.approximation)
            correction = self.traverse_graph(expression.correction)

            if correction is not None:
                # update solution fields with correction
                self.fct_body.append(FieldLoop(solution, [Update(solution, correction)]))

                # apply bc
                self.fct_body.append(ApplyBC(solution))

                # communicate solution fields
                self.fct_body.append(Communicate(solution))

            return solution

        def updateFASApproximation():
            op_restriction = self.traverse_graph(expression.operand1)

            # Store restricted solution in a separate field
            FASApproximation = self.FASApproximation[cur_lvl]
            solution_field = self.solution[cur_lvl]
            loop_stms = []
            loop_stms.append(Assignment(FASApproximation, Multiplication([op_restriction, solution_field])))
            loop_stms.append(Assignment(solution_field, FASApproximation))  # Initialize solution field at coarser level
            self.fct_body.append(FieldLoop(FASApproximation, loop_stms))

            # Communicate fields
            self.fct_body.append(Communicate(FASApproximation))
            self.fct_body.append(Communicate(solution_field))

            return FASApproximation

        def updateRHS(rhs_obj):
            if type(rhs_obj).__name__ != "RightHandSide":  # not the finest grid i.e cur_lvl < max_level
                assert cur_lvl < self.max_level

                # update rhs field
                if self.update_rhs[cur_lvl]:  # rhs at the current level is not updated
                    rhs_expr = self.traverse_graph(rhs_obj)
                    rhs = self.rhs[cur_lvl]
                    self.fct_body.append(FieldLoop(rhs, [Assignment(rhs, rhs_expr)]))
                    self.update_rhs[cur_lvl] = False

        def updateResidual():
            solution = self.solution[cur_lvl]
            op_linear = self.traverse_graph(expression.operator)
            residual = self.residual[cur_lvl]
            rhs = self.rhs[cur_lvl]

            # Update residual fields
            loop_stms = []
            mul_expr = Multiplication([op_linear, solution])
            if self.op_nonlinear is not None:
                op_nonlinear = self.op_nonlinear[cur_lvl]
                mul_expr = Addition([mul_expr, Multiplication([op_nonlinear, solution])])
            loop_stms.append(Assignment(residual,
                                        Subtraction(rhs, mul_expr)))
            self.fct_body.append(FieldLoop(residual, loop_stms))

            # apply bc
            self.fct_body.append(ApplyBC(residual))

            # Communicate fields
            self.fct_body.append(Communicate(residual))

            return residual

        def updateFASerror():
            solution = self.traverse_graph(expression.operand1)
            FASApproximation = self.FASApproximation[cur_lvl]

            # subtract FAS approximation (restricted finer level solution) from solution at current level
            self.fct_body.append(FieldLoop(solution, [Update(solution, FASApproximation, "-=")]))

            # communicate solution field
            self.fct_body.append(Communicate(solution))

            return solution

        def solve():
            rhs_obj = expression.operand2

            # update rhs at coarse grid
            updateRHS(rhs_obj)

            # call coarse grid solver
            self.fct_body.append(FunctionCall(self.fct_CGS + f'@{cur_lvl}'))

            return self.solution[cur_lvl]

        def smoothing():
            residual = expression.operand2
            updateRHS(residual.rhs)

            # call smoother
            self.fct_body.append(FunctionCall(self.fct_smoother + f'@{cur_lvl}'))

        cur_lvl = level(expression)
        expr_type = type(expression).__name__
        if expr_type == "Cycle":
            solution = updateSolution()
            return solution

        elif expr_type == "Multiplication":
            operator = expression.operand1
            operand = expression.operand2
            op_type = type(operator).__name__
            if op_type == "CoarseGridSolver":
                solution = solve()
                return solution
            elif op_type == "Inverse" and self.fct_smoother is not None:  # call predefined smoother
                smoothing()
                return None
            elif op_type == "Operator":
                operand = self.traverse_graph(operand)
                op_linear = self.traverse_graph(operator)
                mul_expr = Multiplication([op_linear, operand])
                if self.op_nonlinear is not None:
                    op_nonlinear = self.op_nonlinear[cur_lvl]
                    mul_expr = Addition([mul_expr, Multiplication([op_nonlinear, operand])])
                return mul_expr
            elif op_type == "Restriction" and ("Approximation" in type(operand).__name__ or type(operand).__name__ == "Cycle"):  # FAS expression
                FASApproximation = updateFASApproximation()
                return FASApproximation
            else:
                operand1 = self.traverse_graph(expression.operand1)
                operand2 = self.traverse_graph(expression.operand2)
                mul_expr = Multiplication([operand1, operand2])

                return mul_expr
        elif expr_type == "Addition":
            operand1 = self.traverse_graph(expression.operand1)
            operand2 = self.traverse_graph(expression.operand2)

            add_expr = Addition([operand1, operand2])

            return add_expr
        elif expr_type == "Subtraction":  # FAS expression
            error = updateFASerror()
            return error

        elif "Residual" in expr_type:
            updateRHS(expression.rhs)
            residual = updateResidual()
            return residual

        elif "Approximation" in expr_type:
            return self.solution[cur_lvl]

        elif expr_type == "RightHandSide":
            return self.rhs[cur_lvl]

        elif expr_type == "Prolongation":
            self.update_rhs[cur_lvl - 1] = True  # rhs at 'cur_lvl-1' needs to be updated
            return self.prolongation[cur_lvl - 1]

        elif expr_type == "Restriction":
            return self.restriction[cur_lvl + 1]

        elif expr_type == "Operator":
            return self.op_linear[cur_lvl]

    def generate_mgfunction(self, expression):
        self.traverse_graph(expression)
        self.fct_mgcycle = Function(self.fct_name, self.fct_body)

    def generate_code(self):
        def create_l4_file():
            # read contents from the template file
            with open(self.build_path + self.exa_file_template, 'r') as f:
                file_contents = f.read()

            # append generated mg cycle and write to output file
            file_contents += "\n" + print_exa(self.fct_mgcycle)
            with open(self.build_path + self.exa_file_out, 'w') as f:
                f.write(file_contents)

            # TODO: modify settings file for each mpi rank

        def modify_settings_file():
            with open(self.build_path + self.settings_file, 'r') as input_file:
                with open(self.build_path + self.settings_file_generated, 'w') as output_file:
                    for line in input_file:
                        tokens = line.split('=')
                        lhs = tokens[0].strip(' \n\t')
                        if lhs == 'configName':
                            config_name = f'{self.problem_name}_{self.mpi_rank}'
                            output_file.write(f'  {lhs}\t = "{config_name}"\n')
                        else:
                            output_file.write(line)

        # create exa4 file with the generated mgcycle function
        create_l4_file()

        # adapt settings file according to mpi rank
        modify_settings_file()

        # generate c++ code with ExaStencils
        subprocess.check_call(["mkdir", "-p", "debug_FAS"], cwd=self.build_path)
        with open(f"{self.build_path}/debug_FAS/generate_output.txt", "w") as f:
            subprocess.check_call(["java", "-cp", self.exastencils_compiler, "Main",
                                   self.settings_file_generated, self.knowledge_file, self.platform_file],
                                  stdout=f, stderr=subprocess.STDOUT, cwd=self.build_path)

    def compile_code(self):
        with open(f"{self.build_path}/debug_FAS/build_output.txt", "w") as f:
            subprocess.check_call(["make"], stdout=f, stderr=subprocess.STDOUT, cwd=self.build_path + "generated/" + f'{self.problem_name}_{self.mpi_rank}')

    def execute_code(self):
        result = subprocess.run(["./exastencils"], stdout=subprocess.PIPE, cwd=self.build_path + "generated/" + f'{self.problem_name}_{self.mpi_rank}')
        return result.stdout.decode('utf8')

    def generate_and_evaluate(self, *args, **kwargs):
        def parse_output(output):
            lines = output.split('\n')
            res_initial = 0.0
            res_final = 0.0
            n = 0
            t = 0.0
            infinity = 1e100
            for line in lines:
                if "Initial Residual" in line:
                    res_initial = float(line.split('Initial Residual:')[1])
                elif "Residual after" in line:
                    res_final = float(line.split('iterations is')[1])
                elif "total no of iterations" in line:
                    n = int(line.split('total no of iterations')[1])
                elif "time to solution (in ms) is" in line:
                    t = float(line.split('time to solution (in ms) is')[1])

            # calculate asymptotic convergence factor
            c = (res_final / res_initial) ** (1.0 / n)
            if math.isinf(c) or math.isnan(c):
                return infinity, infinity, infinity
            else:
                return t, c, n

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

        # create ExaSlang representation of MG expression
        self.fct_body.clear()
        self.generate_mgfunction(expression)

        # generate c++ code (using ExaStencils), and compile executable
        self.generate_code()
        self.compile_code()

        # run executable, and evaluate the outputs
        for i in range(evaluation_samples):
            output = self.execute_code()
            time_solution, convergence_factor, n_iterations = parse_output(output)
            time_solution_list.append(time_solution)
            convergence_factor_list.append(convergence_factor)
            n_iterations_list.append(n_iterations)

        # average evaluated metrics over all samples and return
        return mean(time_solution_list), mean(convergence_factor_list), mean(n_iterations_list)

    # dummy functions to maintain compatibility in the optimisation pipeline
    def generate_storage(self, *args):
        empty_list = []
        return empty_list

    def initialize_code_generation(self, *args):
        pass

    def generate_cycle_function(self, *args):
        expression = None
        for arg in args:
            if type(arg).__name__ == 'Cycle':
                expression = arg

        # create ExaSlang representation of MG expression
        self.fct_body.clear()
        self.generate_mgfunction(expression)

        return print_exa(self.fct_mgcycle)
