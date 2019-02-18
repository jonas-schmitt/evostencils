from evostencils.expressions import base, multigrid as mg, partitioning as part
from evostencils.stencils import constant, periodic
import os
import subprocess
from pathlib import Path
import math


class CycleStorage:
    def __init__(self, level, grid):
        self.level = level
        self.grid = grid
        self.solution = Field(f'Solution', level, self)
        # self.solution_tmp = Field(f'Solution_tmp', level, self)
        self.rhs = Field(f'RHS', level, self)
        # self.rhs_tmp = Field(f'RHS_tmp', level, self)
        self.residual = Field(f'Residual', level, self)
        self.correction = Field(f'Correction', level, self)


class Field:
    def __init__(self, name=None, level=None, cycle_storage=None):
        self.name = name
        self.level = level
        self.cycle_storage = cycle_storage
        self.valid = False

    def to_exa3(self):
        if self.level > 0:
            return f'{self.name}@(finest - {self.level})'
        else:
            return f'{self.name}@finest'


class ProgramGenerator:
    def __init__(self, problem_name: str, exastencils_path: str, op: base.Operator, grid: base.Grid, rhs: base.Grid,
                 identity: base.Identity, interpolation: mg.Interpolation, restriction: mg.Restriction,
                 dimension, coarsening_factor, min_level, max_level, initialization_information, output_path="./execution"):
        self.problem_name = problem_name
        self._exastencils_path = exastencils_path
        self._output_path = output_path
        self._operator = op
        self._grid = grid
        self._rhs = rhs
        self._identity = identity
        self._interpolation = interpolation
        self._restriction = restriction
        self._dimension = dimension
        self._coarsening_factor = coarsening_factor
        self._min_level = min_level
        self._max_level = max_level
        self._initialization_information = initialization_information
        self._compiler_available = False
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if os.path.exists(exastencils_path):
            subprocess.run(['cp', '-r', f'{exastencils_path}/Examples/lib', f'{output_path}/'])
            if os.path.isfile(f'{exastencils_path}/Compiler/Compiler.jar'):
                self._compiler_available = True
        self.generate_settings_file()
        self.generate_knowledge_file()

    @property
    def exastencils_path(self):
        return self._exastencils_path

    @property
    def compiler_available(self):
        return self._compiler_available

    @property
    def output_path(self):
        return self._output_path

    @property
    def operator(self):
        return self._operator

    @property
    def grid(self):
        return self._grid

    @property
    def rhs(self):
        return self._rhs

    @property
    def dimension(self):
        return self._dimension

    @property
    def coarsening_factor(self):
        return self._coarsening_factor

    @property
    def min_level(self):
        return self._min_level

    @property
    def max_level(self):
        return self._max_level

    @property
    def identity(self):
        return self._identity

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def restriction(self):
        return self._restriction

    def generate_settings_file(self):
        tmp = f'user\t= "Guest"\n\n'
        tmp += f'basePathPrefix\t= "{self.output_path}"\n\n'
        tmp += f'l3file\t= "{self.problem_name}.exa3"\n\n'
        tmp += f'debugL3File\t= "Debug/{self.problem_name}_debug.exa3"\n'
        tmp += f'debugL4File\t= "Debug/{self.problem_name}_debug.exa4"\n\n'
        tmp += f'htmlLogFile\t= "Debug/{self.problem_name}_log.html"\n\n'
        tmp += f'outputPath\t= "generated/{self.problem_name}"\n\n'
        tmp += f'produceHtmlLog\t= true\n'
        tmp += f'timeStrategies\t= true\n\n'
        tmp += f'buildfileGenerators\t= {{"MakefileGenerator"}}\n'
        with open(f'{self.output_path}/{self.problem_name}.settings', "w") as file:
            print(tmp, file=file)

    def generate_knowledge_file(self, discretization_type="FiniteDifferences",
                                domain="domain_onePatch", parallelization="parallelization_pureOmp"):
        tmp = f'dimensionality\t= {self.dimension}\n\n'
        tmp += f'minLevel\t= {self.min_level}\n'
        tmp += f'maxLevel\t= {self.max_level}\n\n'
        tmp += f'discr_type\t= "{discretization_type}"\n\n'
        tmp += f'import "lib/{domain}.knowledge"\n'
        tmp += f'import "lib/{parallelization}.knowledge"\n'
        with open(f'{self.output_path}/{self.problem_name}.knowledge', "w") as file:
            print(tmp, file=file)

    def generate_boilerplate(self, storages, dimension, epsilon=1e-10, level=0):
        if dimension == 1:
            program = "Domain global < [0.0] to [1.0] >\n"
        elif dimension == 2:
            program = "Domain global < [0.0, 0.0] to [1.0, 1.0] >\n"
        elif dimension == 3:
            program = "Domain global < [0.0, 0.0, 0.0] to [1.0, 1.0, 1.0] >\n"
        else:
            raise RuntimeError("Only 1-3D currently supported")
        program += self.add_field_declarations_to_program_string(storages, level)
        program += '\n'
        program += self.add_operator_declarations_to_program_string()
        program += '\n'
        program += self.generate_solver_function("gen_solve", storages, epsilon=epsilon, level=level)
        program += "\nApplicationHint {\n\tl4_genDefaultApplication = true\n}\n"
        return program

    def generate_cycle_function(self, expression, storages):
        base_level = 0
        for i, storage in enumerate(storages):
            if expression.grid.size == storage.grid.size:
                expression.storage = storage.solution
                base_level = i
                break
        self.assign_storage_to_subexpressions(expression, storages, base_level)
        program = ''
        program += f'Function Cycle@(finest - {base_level}) {{\n'
        program += self.generate_multigrid(expression, storages)
        program += f'}}\n'
        return program

    def write_program_to_file(self, program: str):
        with open(f'{self.output_path}/{self.problem_name}.exa3', "w") as file:
            print(program, file=file)

    def run_exastencils_compiler(self, platform='linux'):
        import subprocess
        result = subprocess.run(['java', '-cp',
                                 f'{self.exastencils_path}/Compiler/Compiler.jar', 'Main',
                                 f'{self.output_path}/{self.problem_name}.settings',
                                 f'{self.output_path}/{self.problem_name}.knowledge',
                                 f'{self.output_path}/lib/{platform}.platform'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not result.returncode == 0:
            return result.returncode

        result = subprocess.run(['make', '-j4', '-s', '-C', f'{self.output_path}/generated/{self.problem_name}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.returncode

    def execute(self, platform='linux', infinity=1e100):
        result = subprocess.run(['java', '-cp',
                                 f'{self.exastencils_path}/Compiler/Compiler.jar', 'Main',
                                 f'{self.output_path}/{self.problem_name}.settings',
                                 f'{self.output_path}/{self.problem_name}.knowledge',
                                 f'{self.output_path}/lib/{platform}.platform'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not result.returncode == 0:
            return infinity, infinity
        result = subprocess.run(['make', '-j4', '-s', '-C', f'{self.output_path}/generated/{self.problem_name}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not result.returncode == 0:
            return infinity, infinity
        total_time = 0
        sum_of_convergence_factors = 0
        number_of_samples = 10
        for i in range(number_of_samples):
            result = subprocess.run([f'{self.output_path}/generated/{self.problem_name}/exastencils'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not result.returncode == 0:
                return infinity, infinity
            output = result.stdout.decode('utf8')
            time_to_solution, convergence_factor = self.parse_output(output)
            if math.isinf(convergence_factor) or math.isnan(convergence_factor) or not convergence_factor < 1:
                return infinity, infinity
            total_time += time_to_solution
            sum_of_convergence_factors += convergence_factor
        return total_time / number_of_samples, sum_of_convergence_factors / number_of_samples

    @staticmethod
    def parse_output(output: str):
        lines = output.splitlines()
        convergence_factor = 1
        count = 0
        for line in lines:
            if 'convergence factor' in line:
                tmp = line.split('convergence factor is ')
                convergence_factor *= float(tmp[-1])
                count += 1
        convergence_factor = math.pow(convergence_factor, 1/count)
        tmp = lines[-1].split(' ')
        time_to_solution = float(tmp[-2])
        return time_to_solution, convergence_factor

    @staticmethod
    def obtain_coarsest_level(cycle: mg.Cycle, base_level=0) -> int:
        def recursive_descent(expression: base.Expression, current_size: tuple, current_level: int):
            if isinstance(expression, mg.Cycle):
                if expression.grid.size < current_size:
                    new_size = expression.grid.size
                    new_level = current_level + 1
                else:
                    new_size = current_size
                    new_level = current_level
                level_iterate = recursive_descent(expression.iterate, new_size, new_level)
                level_correction = recursive_descent(expression.correction, new_size, new_level)
                return max(level_iterate, level_correction)
            elif isinstance(expression, mg.Residual):
                level_iterate = recursive_descent(expression.iterate, current_size, current_level)
                level_rhs = recursive_descent(expression.rhs, current_size, current_level)
                return max(level_iterate, level_rhs)
            elif isinstance(expression, base.BinaryExpression):
                level_operand1 = recursive_descent(expression.operand1, current_size, current_level)
                level_operand2 = recursive_descent(expression.operand2, current_size, current_level)
                return max(level_operand1, level_operand2)
            elif isinstance(expression, base.UnaryExpression):
                return recursive_descent(expression.operand, current_size, current_level)
            elif isinstance(expression, base.Scaling):
                return recursive_descent(expression.operand, current_size, current_level)
            elif isinstance(expression, base.Entity):
                return current_level
            else:
                raise RuntimeError("Unexpected expression")
        return recursive_descent(cycle, cycle.grid.size, base_level) + 1

    def generate_storage(self, maximum_level):
        tmps = []
        grid = self.grid
        for level in range(0, maximum_level + 1):
            tmps.append(CycleStorage(level, grid))
            grid = mg.get_coarse_grid(grid, self.coarsening_factor)
        return tmps

    @staticmethod
    def needs_storage(expression: base.Expression):
        return expression.shape[1] == 1

    @staticmethod
    def adjust_storage_index(node, storages, i) -> int:
        if node.grid.size > storages[i].grid.size:
            return i-1
        elif node.grid.size < storages[i].grid.size:
            return i+1
        else:
            return i

    # Warning: This function modifies the expression passed to it
    @staticmethod
    def assign_storage_to_subexpressions(node: base.Expression, storages: [CycleStorage], i: int):
        if isinstance(node, mg.Cycle):
            i = ProgramGenerator.adjust_storage_index(node, storages, i)
            node.iterate.storage = storages[i].solution
            node.correction.storage = storages[i].correction
            ProgramGenerator.assign_storage_to_subexpressions(node.correction, storages, i)
        elif isinstance(node, mg.Residual):
            i = ProgramGenerator.adjust_storage_index(node, storages, i)
            node.iterate.storage = storages[i].solution
            node.rhs.storage = storages[i].rhs
            ProgramGenerator.assign_storage_to_subexpressions(node.iterate, storages, i)
            ProgramGenerator.assign_storage_to_subexpressions(node.rhs, storages, i)
        elif isinstance(node, base.BinaryExpression):
            operand1 = node.operand1
            operand2 = node.operand2
            if ProgramGenerator.needs_storage(operand2):
                i = ProgramGenerator.adjust_storage_index(operand2, storages, i)
                if isinstance(operand2, mg.Residual):
                    operand2.storage = storages[i].residual
                else:
                    operand2.storage = storages[i].correction
            ProgramGenerator.assign_storage_to_subexpressions(operand1, storages, i)
            ProgramGenerator.assign_storage_to_subexpressions(operand2, storages, i)
        elif isinstance(node, base.UnaryExpression) or isinstance(node, base.Scaling):
            operand = node.operand
            if ProgramGenerator.needs_storage(operand):
                i = ProgramGenerator.adjust_storage_index(operand, storages, i)
                operand.storage = storages[i].correction
        elif isinstance(node, base.RightHandSide):
            i = ProgramGenerator.adjust_storage_index(node, storages, i)
            node.storage = storages[i].rhs
        elif isinstance(node, base.Grid) or isinstance(node, base.ZeroGrid):
            i = ProgramGenerator.adjust_storage_index(node, storages, i)
            node.storage = storages[i].solution

    def add_field_declarations_to_program_string(self, storages: [CycleStorage], level=0):
        program_string = ''
        for i, storage in enumerate(storages):
            solution = storage.solution.to_exa3()
            # solution_tmp = storage.solution_tmp.to_exa3()
            rhs = storage.rhs.to_exa3()
            # rhs_tmp = storage.rhs_tmp.to_exa3()
            residual = storage.residual.to_exa3()
            correction = storage.correction.to_exa3()
            program_string += f'Field {solution} with Real on Node of global = 0.0\n'
            if i == level:
                program_string += f'Field {solution} on boundary = ' \
                                  f'{self._initialization_information.get_boundary_as_str()}\n'
            else:
                program_string += f'Field {solution} on boundary = 0.0\n'

            if i == level:
                program_string += f'Field {rhs} with Real on Node of global = ' \
                                f'{self._initialization_information.get_rhs_as_str()}\n'
            else:
                program_string += f'Field {rhs} with Real on Node of global = 0\n'

            program_string += f'Field {rhs} on boundary = 0.0\n'

            program_string += f'Field {residual} with Real on Node of global = 0.0\n'
            program_string += f'Field {residual} on boundary = 0.0\n'

            program_string += f'Field {correction} with Real on Node of global = 0.0\n'
            program_string += f'Field {correction} on boundary = 0.0\n'
        return program_string

    @staticmethod
    def add_constant_stencil_to_program_string(program_string: str, level: int, operator_name: str,
                                               stencil: constant.Stencil):
        program_string += f"Operator {operator_name}@(finest - {level}) from Stencil {{\n"
        indent = '\t'
        for offset, value in stencil.entries:
            program_string += indent
            program_string += '['
            for i, _ in enumerate(offset):
                program_string += f'i{i}'
                if i < len(offset) - 1:
                    program_string += ', '
            program_string += f'] from ['
            for i, o in enumerate(offset):
                if o == 0:
                    program_string += f'i{i}'
                else:
                    program_string += f'(i{i} + ({o}))'
                if i < len(offset) - 1:
                    program_string += ', '
            program_string += f'] with {value}\n'
        program_string += '}\n'
        return program_string

    def add_operator_declarations_to_program_string(self):
        program_string = ''
        program_string += self.operator.generate_exa3()
        program_string += self.identity.generate_exa3()
        program_string += self.interpolation.generate_exa3()
        program_string += self.restriction.generate_exa3()
        return program_string

    @staticmethod
    def determine_operator_level(operator, storages) -> int:
        for storage in storages:
            if operator.grid.size == storage.grid.size:
                return storage.level
        raise RuntimeError("No fitting level")

    def generate_coarse_grid_solver(self, expression: base.Expression, storages: [CycleStorage]):
        # TODO replace hard coded RB-GS by generic implementation
        i = expression.storage.level
        n = 1000
        program = f'\t{storages[i].solution.to_exa3()} = 0\n'
        program += f'\trepeat {n} times {{\n'
        # TODO currently hard coded for two dimensions
        program += f'\t\t{storages[i].solution.to_exa3()} ' \
                   f'+= ((0.8 * diag_inv({self.operator.name}@(finest - {i}))) * ' \
                   f'({expression.storage.to_exa3()} - ({self.operator.name}@(finest - {i}) * ' \
                   f'{storages[i].solution.to_exa3()}))) '
        program += f'where (((i0'
        for j in range(1, expression.grid.dimension):
            program += f' + i{j}'
        program += f') % 2) == 0)\n'
        program += f'\t\t{storages[i].solution.to_exa3()} ' \
                   f'+= ((0.8 * diag_inv({self.operator.name}@(finest - {i}))) * ' \
                   f'({expression.storage.to_exa3()} - ({self.operator.name}@(finest - {i}) * ' \
                   f'{storages[i].solution.to_exa3()}))) '
        program += f'where (((i0'
        for j in range(1, expression.grid.dimension):
            program += f' + i{j}'
        program += f') % 2) == 1)\n'
        program += '\t}\n'
        program += f'\t{expression.storage.to_exa3()} = {storages[i].solution.to_exa3()}\n'
        return program

    def generate_multigrid(self, expression: base.Expression, storages) -> str:
        # import decimal
        # if expression.program is not None:
        #     return expression.program
        program = ''
        if expression.storage is not None:
            expression.storage.valid = False

        if isinstance(expression, mg.Cycle):
            field = expression.storage
            # decimal.getcontext().prec = 14
            # weight = decimal.Decimal(expression.weight)
            weight = expression.weight
            if isinstance(expression.correction, base.Multiplication) \
                    and part.can_be_partitioned(expression.correction.operand1):
                new_iterate_str = expression.storage.to_exa3()
                iterate_str = expression.iterate.storage.to_exa3()
                stencil = expression.correction.operand1.generate_stencil()
                if isinstance(expression.correction.operand2, mg.Residual) and periodic.is_diagonal(stencil):
                    residual = expression.correction.operand2
                    program += self.generate_multigrid(residual.iterate, storages)
                    if not residual.rhs.storage.valid:
                        program += self.generate_multigrid(residual.rhs, storages)
                        residual.rhs.storage.valid = True

                    if isinstance(residual.iterate, base.ZeroGrid):
                        program += f'\t{expression.iterate.storage.to_exa3()} = 0\n'
                    if isinstance(expression.correction.operand1, base.Operator):
                        operator_str = f'diag({self.generate_multigrid(expression.correction.operand1, storages)})'
                    else:
                        operator_str = f'({self.generate_multigrid(expression.correction.operand1, storages)})'
                    correction_str = f'({residual.rhs.storage.to_exa3()} - ' \
                                     f'{self.generate_multigrid(residual.operator, storages)} * ' \
                                     f'{residual.iterate.storage.to_exa3()})'
                else:
                    program += self.generate_multigrid(expression.correction.operand2, storages)
                    if isinstance(expression.correction.operand1, base.BinaryExpression) or isinstance(expression.correction.operand1, base.Scaling):
                        operator_str = f'({self.generate_multigrid(expression.correction.operand1, storages)})'
                    else:
                        operator_str = f'{self.generate_multigrid(expression.correction.operand1, storages)}'
                    correction_str = f'{expression.correction.operand2.storage.to_exa3()}'

                smoother = f'{new_iterate_str} = {iterate_str} + ({weight}) ' \
                           f'* {operator_str} * {correction_str}'
                if expression.partitioning == part.RedBlack:
                    program += f'\t{smoother} where (((i0'
                    for i in range(1, expression.grid.dimension):
                        program += f' + i{i}'
                    program += f') % 2) == 0)\n'
                    program += f'\t{smoother} where (((i0'
                    for i in range(1, expression.grid.dimension):
                        program += f' + i{i}'
                    program += f') % 2) == 1)\n'
                elif expression.partitioning == part.Single:
                    program += f'\t{smoother}\n'
                else:
                    raise RuntimeError(f"Partitioning {expression.partitioning} not supported")

            else:
                program += self.generate_multigrid(expression.correction, storages)
                program += f'\t{field.to_exa3()} = ' \
                           f'{expression.iterate.storage.to_exa3()} + ' \
                           f'({weight}) * {expression.correction.storage.to_exa3()}\n'
        elif isinstance(expression, mg.Residual):
            program += self.generate_multigrid(expression.iterate, storages)
            if not expression.rhs.storage.valid:
                program += self.generate_multigrid(expression.rhs, storages)
                expression.rhs.storage.valid = True

            if isinstance(expression.iterate, base.ZeroGrid):
                program += f'\t{expression.iterate.storage.to_exa3()} = 0\n'
            program += f'\t{expression.storage.to_exa3()} = {expression.rhs.storage.to_exa3()} - ' \
                       f'{self.generate_multigrid(expression.operator, storages)} * ' \
                       f'{expression.iterate.storage.to_exa3()}\n'
        elif isinstance(expression, base.BinaryExpression):
            if isinstance(expression, base.Multiplication):
                operator_str = "*"
            elif isinstance(expression, base.Addition):
                operator_str = "+"
            elif isinstance(expression, base.Subtraction):
                operator_str = "-"
            else:
                raise RuntimeError("Unexpected branch")

            if expression.storage is None:
                return f'{self.generate_multigrid(expression.operand1, storages)} ' \
                       f'{operator_str} ' \
                       f'{self.generate_multigrid(expression.operand2, storages)}'
            else:
                operand1 = expression.operand1
                operand2 = expression.operand2
                if expression.operand1.storage is None:
                    if isinstance(operand1, mg.CoarseGridSolver):
                        program += self.generate_multigrid(operand2, storages)
                        if operand1.expression is not None:
                            level = expression.operand2.storage.level
                            program += f'\t{storages[level].solution.to_exa3()} = 0\n'
                            program += f'\t{storages[level].rhs.to_exa3()} = {expression.operand2.storage.to_exa3()}\n'
                            program += f'\tCycle@(finest - {level})()\n'
                            program += f'\t{expression.storage.to_exa3()} = {storages[level].solution.to_exa3()}\n'
                        else:
                            if expression.storage is not expression.operand2.storage:
                                program += f'\t{expression.storage.to_exa3()} = {expression.operand2.storage.to_exa3()}\n'
                            program += self.generate_coarse_grid_solver(expression, storages)
                        return program
                    else:
                        tmp1 = f'({self.generate_multigrid(operand1, storages)})'
                else:
                    program += self.generate_multigrid(operand1, storages)
                    tmp1 = f'{operand1.storage.to_exa3()}'

                if expression.operand2.storage is None:
                    tmp2 = self.generate_multigrid(operand2, storages)
                else:
                    program += self.generate_multigrid(operand2, storages)
                    tmp2 = f'{operand2.storage.to_exa3()}'
                program += f'\t{expression.storage.to_exa3()} = {tmp1} {operator_str} {tmp2}\n'
        elif isinstance(expression, base.Scaling):
            program = ""
            operand = expression.operand
            field = expression.storage
            if field is None:
                tmp = self.generate_multigrid(expression.operand, storages)
            else:
                program += self.generate_multigrid(expression.operand, storages)
                tmp = f'{operand.storage.to_exa()}'
            program += f'({expression.factor}) * {tmp}'
        elif isinstance(expression, base.Inverse):
            if isinstance(expression.operand, base.Inverse) or isinstance(expression.operand, base.Identity):
                program = self.generate_multigrid(expression.operand, storages)
            else:
                program = f'inverse({self.generate_multigrid(expression.operand, storages)})'
        elif isinstance(expression, base.Diagonal):
            program = f'diag({self.generate_multigrid(expression.operand, storages)})'
        elif isinstance(expression, mg.Restriction):
            level_offset = self.determine_operator_level(expression, storages) - 1
            program = f'{expression.name}@(finest - {level_offset})'
        elif isinstance(expression, mg.Interpolation):
            level_offset = self.determine_operator_level(expression, storages) + 1
            program = f'{expression.name}@(finest - {level_offset})'
        elif isinstance(expression, base.Identity):
            program = f'1'
        elif isinstance(expression, base.Operator):
            program = f'{expression.name}@(finest - {self.determine_operator_level(expression, storages)})'
        elif isinstance(expression, base.Grid):
            pass
            # program = f'{expression.storage.to_exa3()}'
        else:
            print(type(expression))
            raise NotImplementedError("Case not implemented")
        expression.program = program
        return program

    @staticmethod
    def generate_l2_norm_residual(function_name: str, storages: [CycleStorage], level=0) -> str:
        residual = storages[level].residual.to_exa3()
        program = ""
        program += f"Function {function_name}@(finest - {level}) : Real {{\n"
        program += f"\tVar gen_resNorm : Real = 0.0\n"
        program += f"\tgen_resNorm += dot ( {residual}, {residual} )\n"
        program += f"\treturn sqrt ( gen_resNorm )\n"
        program += f"}}\n"
        return program

    def generate_solver_function(self, function_name: str, storages: [CycleStorage], epsilon=1e-10,
                                 maximum_number_of_iterations=100, level=0) -> str:
        assert maximum_number_of_iterations >= 1, "At least one iteration required"
        operator = f"{self.operator.name}@(finest - {level})"
        solution = storages[level].solution.to_exa3()
        rhs = storages[level].rhs.to_exa3()
        residual = storages[level].residual.to_exa3()
        compute_residual_norm = "gen_resNorm"
        program = self.generate_l2_norm_residual(compute_residual_norm, storages, level)
        program += f"\nFunction {function_name}@finest {{\n"
        program += f"\t{residual} = ({rhs} - ({operator} * {solution}))\n"
        program += f"\tVar gen_initRes: Real = {compute_residual_norm}@(finest - {level})()\n"
        program += f"\tVar gen_curRes: Real = gen_initRes\n"
        program += f"\tVar gen_prevRes: Real = gen_curRes\n"
        program += f'\tprint("Starting residual:", gen_initRes)\n'
        program += f"\tVar gen_curIt : Integer = 0\n"
        program += f"\trepeat until(((gen_curIt >= {maximum_number_of_iterations}) " \
                   f"|| (gen_curRes <= ({epsilon} * gen_initRes))) || (gen_curRes <= 0.0)) {{\n"
        program += f"\t\tgen_curIt += 1\n" \
                   f"\t\tCycle@(finest - {level})()\n"
        program += f"\t\t{residual} = ({rhs} - ({operator} * {solution}))\n"
        program += f"\t\tgen_prevRes = gen_curRes\n"
        program += f"\t\tgen_curRes = {compute_residual_norm}@(finest - {level})()\n"
        program += f'\t\tprint("Residual after", gen_curIt, "iterations is", gen_curRes, ' \
                   f'"--- convergence factor is", (gen_curRes / gen_prevRes))'
        program += "\t}\n"
        program += "}\n"
        return program

    @staticmethod
    def invalidate_storages(storages: [CycleStorage]):
        for storage in storages:
            storage.residual.valid = False
            storage.rhs.valid = False
            storage.solution.valid = False
            storage.correction.valid = False

