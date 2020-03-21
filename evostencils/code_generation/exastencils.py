from evostencils.expressions import base, partitioning as part, system, transformations
from evostencils.expressions import krylov_subspace
from evostencils.initialization import multigrid, parser
import os
import subprocess
import math
import sympy
import time
from typing import List


class CycleStorage:
    def __init__(self, equations: [multigrid.EquationInfo], fields: [sympy.Symbol], grids: List[base.Grid]):
        self.grid = grids
        self.solution = [Field(f'{symbol.name}', g.level, self) for g, symbol in zip(grids, fields)]
        self.rhs = [Field(f'{eq_info.rhs_name}', g.level, self) for g, eq_info in zip(grids, equations)]
        self.residual = [Field(f'gen_residual_{symbol.name}', g.level, self) for g, symbol in zip(grids, fields)]
        self.correction = [Field(f'gen_error_{symbol.name}', g.level, self) for g, symbol in zip(grids, fields)]


class Field:
    def __init__(self, name=None, level=None, cycle_storage=None):
        self.name = name
        self.level = level
        self.cycle_storage = cycle_storage

    def to_exa(self):
        return f'{self.name}@{self.level}'


class ProgramGenerator:
    def __init__(self, absolute_compiler_path: str, base_path: str, settings_path: str, knowledge_path: str,
                 mpi_rank=0, platform='linux'):
        self._average_generation_time = 0
        self._counter = 0
        self.timeout_copy_file = 60
        self.timeout_evaluate = 600
        self.timeout_exastencils_compiler = 300
        self.timeout_c_compiler = 180
        self._absolute_compiler_path = absolute_compiler_path
        self._base_path = base_path
        self._knowledge_path = knowledge_path
        self._settings_path = settings_path
        self._dimension, self._min_level, self._max_level = \
            parser.extract_knowledge_information(base_path, knowledge_path)
        self._base_path_prefix, self._problem_name, self._debug_l3_path, self._output_path = \
            parser.extract_settings_information(base_path, settings_path)
        self._mpi_rank = mpi_rank
        self._platform = platform
        self._knowledge_path_generated = f'{self._base_path_prefix}/{self.problem_name}_{self.mpi_rank}.knowledge'
        self._settings_path_generated = f'{self._base_path_prefix}/{self.problem_name}_{self.mpi_rank}.settings'
        self._layer3_path_generated = f'{self._base_path_prefix}/{self.problem_name}_{self.mpi_rank}.exa3'
        self._output_path_generated = None
        self.run_exastencils_compiler()
        self._equations, self._operators, self._fields = \
            parser.extract_l2_information(f'{base_path}/{self._debug_l3_path}', self.dimension)
        size = 2 ** self._max_level
        grid_size = tuple([size] * self.dimension)
        h = 1 / (2 ** self._max_level)
        step_size = tuple([h] * self.dimension)
        tmp = tuple([2] * self.dimension)
        self._coarsening_factor = [tmp for _ in range(len(self.fields))]
        self._finest_grid = [base.Grid(grid_size, step_size, self.max_level) for _ in range(len(self.fields))]
        self._compiler_available = False
        if os.path.exists(absolute_compiler_path) and os.path.isfile(absolute_compiler_path):
            self._compiler_available = True
        else:
            raise RuntimeError("Compiler not found. Aborting.")
        self._solver_cache = {}
        self._field_declaration_cache = set()

    @property
    def absolute_compiler_path(self):
        return self._absolute_compiler_path

    @property
    def knowledge_path(self):
        return self._knowledge_path

    @property
    def settings_path(self):
        return self._settings_path

    @property
    def problem_name(self):
        return self._problem_name

    @property
    def compiler_available(self):
        return self._compiler_available

    @property
    def base_path(self):
        return self._base_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def platform(self):
        return self._platform

    @property
    def dimension(self):
        return self._dimension

    @property
    def finest_grid(self):
        return self._finest_grid

    @property
    def equations(self):
        return self._equations

    @property
    def operators(self):
        return self._operators

    @property
    def fields(self):
        return self._fields

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
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def knowledge_path_generated(self):
        return self._knowledge_path_generated

    @property
    def settings_path_generated(self):
        return self._settings_path_generated

    @staticmethod
    def generate_global_weights(n: int):
        # Hack to change the weights after generation
        program = 'Globals {\n'
        for i in range(0, n):
            program += f'\tVar omega_{i} : Real = 1.0\n'
        program += '}\n'
        return program

    def generate_global_weight_initializations(self, output_path, weights: List[float]):
        # Hack to change the weights after generation
        weights = reversed(weights)
        path_to_file = f'{self.base_path}/{output_path}/Global/Global_initGlobals.cpp'
        subprocess.run(['cp', path_to_file, f'{path_to_file}.backup'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout_copy_file)
        with open(path_to_file, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1]
            lines = lines[:-1]
        content = ''
        for line in lines:
            content += line
        for i, weight in enumerate(weights):
            lines.append(f'\tomega_{i} = {weight};\n')
            content += lines[-1]
        content += last_line
        with open(path_to_file, 'w') as file:
            file.write(content)

    def restore_global_initializations(self, output_path):
        # Hack to change the weights after generation
        path_to_file = f'{self.base_path}/{output_path}/Global/Global_initGlobals.cpp'
        subprocess.run(['cp', f'{path_to_file}.backup', path_to_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout_copy_file)

    @staticmethod
    def get_solution_field(storages: List[CycleStorage], index: int, level: int, max_level: int):
        offset = storages[0].grid[index].level - level
        if level < max_level:
            return storages[offset].correction[index]
        else:
            return storages[offset].solution[index]

    @staticmethod
    def get_rhs_field(storages: List[CycleStorage], index: int, level: int):
        offset = storages[0].grid[index].level - level
        return storages[offset].rhs[index]

    @staticmethod
    def get_residual_field(storages: List[CycleStorage], index: int, level: int):
        offset = storages[0].grid[index].level - level
        return storages[offset].residual[index]

    @staticmethod
    def get_correction_field(storages: List[CycleStorage], index: int, level: int):
        offset = storages[0].grid[index].level - level
        return storages[offset].correction[index]

    def generate_cycle_function(self, expression: base.Expression, storages: List[CycleStorage], min_level: int, level,
                                max_level: int, use_global_weights=False):
        program = f'Function gen_mgCycle@{level} {{\n'
        program += self.generate_multigrid(expression, storages, min_level, max_level, use_global_weights)
        program += '}\n\n'
        for key, value in self._solver_cache.items():
            solver_function = value[0]
            valid = value[1]
            if valid:
                program += solver_function + '\n'
        self.invalidate_solver_cache()

        def restore_valid_flag(expr: base.Expression):
            if expr is not None:
                expr.valid = False
                expr.mutate(restore_valid_flag)
        restore_valid_flag(expression)

        return program

    def run_exastencils_compiler(self, knowledge_path=None, settings_path=None):
        if knowledge_path is None:
            knowledge_path = self.knowledge_path
        if settings_path is None:
            settings_path = self.settings_path
        current_path = os.getcwd()
        os.chdir(self.base_path)
        if self._counter == 0:
            timeout = self.timeout_exastencils_compiler
        else:
            timeout = 10 * self._average_generation_time
            if self._counter < 5:
                timeout = 1.5 * timeout - self._counter * self._average_generation_time
        try:
            timeout = min(self.timeout_exastencils_compiler, timeout)
            result = subprocess.run(['java', '-cp',
                                     self.absolute_compiler_path, 'Main',
                                     f'{self.base_path}/{settings_path}',
                                     f'{self.base_path}/{knowledge_path}',
                                     f'{self.base_path}/lib/{self.platform}.platform'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise e
        os.chdir(current_path)
        if result.returncode != 0:
            raise RuntimeError("Compiler not working. Aborting.")
        return result.returncode

    def run_c_compiler(self, makefile_path):
        result = subprocess.run(['make', '-j4', '-s', '-C', f'{self.base_path}/{makefile_path}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout_c_compiler)
        return result.returncode

    def evaluate(self, executable_path, infinity=1e300, number_of_samples=1):
        total_time = 0
        sum_of_convergence_factors = 0
        number_of_iterations = None
        for i in range(number_of_samples):
            result = subprocess.run([f'{self.base_path}/{executable_path}/exastencils'],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=self.timeout_evaluate)
            if not result.returncode == 0:
                return infinity, infinity, infinity
            output = result.stdout.decode('utf8')
            time_to_solution, convergence_factor, number_of_iterations = self.parse_output(output)
            if math.isinf(convergence_factor) or math.isnan(convergence_factor):
                return infinity, infinity, infinity
            total_time += time_to_solution
            sum_of_convergence_factors += convergence_factor
        return total_time / number_of_samples, sum_of_convergence_factors / number_of_samples, number_of_iterations

    def initialize_code_generation(self, min_level: int, max_level: int, iteration_limit=100):
        knowledge_path = self.generate_level_adapted_knowledge_file(min_level, max_level)
        self.generate_adapted_layer_files(iteration_limit)
        settings_path = self.generate_adapted_settings_file(l2file_required=True)
        if self._counter == 0:
            start_time = time.time()
            self.run_exastencils_compiler(knowledge_path=knowledge_path, settings_path=settings_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self._counter += 1
            self._average_generation_time += (elapsed_time - self._average_generation_time) / self._counter
        else:
            self.run_exastencils_compiler(knowledge_path=knowledge_path, settings_path=settings_path)
        settings_path = self.generate_adapted_settings_file()
        _, __, ___, output_path_generated = \
            parser.extract_settings_information(self.base_path, settings_path)
        self._output_path_generated = output_path_generated
        debug_l3_path = f'{self.base_path}/{self._debug_l3_path}'.replace('_debug.exa3', f'_{self.mpi_rank}_debug.exa3')
        l3_path = f'{self.base_path}/{self._base_path_prefix}/{self.problem_name}_base_{self.mpi_rank}.exa3'
        subprocess.run(['cp', debug_l3_path, l3_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path_generated

    def generate_and_evaluate(self, expression: base.Expression, storages: List[CycleStorage], min_level: int,
                              max_level: int, solver_program: str,
                              infinity=1e300, number_of_samples=1):
        cycle_function = self.generate_cycle_function(expression, storages, min_level, max_level, self.max_level)
        self.generate_l3_file(min_level, self.max_level, solver_program + cycle_function)
        try:
            start_time = time.time()
            returncode = self.run_exastencils_compiler(knowledge_path=self.knowledge_path_generated,
                                                       settings_path=self.settings_path_generated)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if returncode != 0:
                return infinity, infinity, infinity
        except subprocess.TimeoutExpired:
            return infinity, infinity, infinity
        self._counter += 1
        self._average_generation_time += (elapsed_time - self._average_generation_time) / self._counter
        if self._output_path_generated is None:
            raise RuntimeError('Output path not set')
        returncode = self.run_c_compiler(self._output_path_generated)
        if returncode != 0:
            return infinity, infinity, infinity
        runtime, convergence_factor, number_of_iterations = self.evaluate(self._output_path_generated,
                                                                          infinity, number_of_samples)
        return runtime, convergence_factor, number_of_iterations

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
        number_of_iterations = len(lines) - 3
        return time_to_solution, convergence_factor, number_of_iterations

    def generate_storage(self, min_level: int, max_level: int, finest_grids: List[base.Grid]):
        storage = []
        grid = finest_grids
        for i in range(min_level, max_level+1):
            storage.append(CycleStorage(self.equations, self.fields, grid))
            grid = system.get_coarse_grid(grid, self.coarsening_factor)
        return storage

    def obtain_correct_source_field(self, expression: base.Expression, storages: List[CycleStorage], index: int, level: int,
                                    max_level: int):
        if isinstance(expression, system.Approximation) or isinstance(expression, base.Cycle):
            solution_field = self.get_solution_field(storages, index, level, max_level)
            return solution_field
        elif isinstance(expression, base.Residual):
            return self.get_residual_field(storages, index, level)
        elif isinstance(expression, system.RightHandSide):
            return self.get_rhs_field(storages, index, level)
        else:
            return self.get_correction_field(storages, index, level)

    # TODO add type annotations
    @staticmethod
    def generate_solve_locally(key, value, indentation, max_level):
        program = ''
        rhs = f'{value[1]}@['
        if key[1] < max_level:
            unknown = f'gen_error_'
        else:
            unknown = ''
        unknown += f'{key[0]}@{key[1]}@['
        for offset in key[2][:-1]:
            unknown += f'{offset}, '
            rhs += f'{offset}, '
        unknown += f'{key[2][-1]}]'
        rhs += f'{key[2][-1]}]'
        transformed_equation = value[0]
        for symbol in value[0].free_symbols:
            tokens = symbol.name.split('_')
            if tokens[-1] == 'new':
                transformed_equation = transformed_equation.subs(symbol, sympy.Symbol(tokens[0]))
        if key[1] < max_level:
            tmp = transformed_equation
            for symbol in transformed_equation.free_symbols:
                tmp = tmp.subs(symbol, sympy.Symbol(f'gen_error_{symbol.name}'))
            transformed_equation = tmp
        program += f'\t\t{indentation}{unknown} => ({transformed_equation}) == {rhs}\n'
        return program

    def generate_sympy_expression_for_operator_entry(self, expression, level):
        if isinstance(expression, base.BinaryExpression):
            subterm1 = self.generate_sympy_expression_for_operator_entry(expression.operand1, level)
            subterm2 = self.generate_sympy_expression_for_operator_entry(expression.operand2, level)
            if isinstance(expression, base.Multiplication):
                return subterm1 * subterm2
            elif isinstance(expression, base.Addition):
                return subterm1 + subterm2
            elif isinstance(expression, base.Subtraction):
                return subterm1 - subterm2
            else:
                raise RuntimeError("Invalid expression")
        elif isinstance(expression, base.Scaling):
            subterm = self.generate_sympy_expression_for_operator_entry(expression.operand, level)
            return expression.factor * subterm
        elif isinstance(expression, base.Operator):
            if isinstance(expression, base.Identity):
                return sympy.sympify(1)
            else:
                return sympy.MatrixSymbol(f'{expression.name}@{level}', expression.shape[0], expression.shape[1])
                # return sympy.Symbol(f'{expression.name}@{level}')
        else:
            raise RuntimeError("Invalid expression")

    def generate_multigrid(self, expression: base.Expression, storages: List[CycleStorage], min_level: int,
                           max_level: int, use_global_weights=False):
        program = ''
        if isinstance(expression, base.Cycle):
            weight = expression.relaxation_factor
            # Hack to change the weights after generation

            if use_global_weights and hasattr(expression, 'global_id'):
                if expression.global_id is None:
                    weight = '1'
                else:
                    weight = f'omega_{expression.global_id}'

            correction = expression.correction
            if isinstance(correction, base.Residual):
                if not isinstance(expression.correction.rhs, system.RightHandSide) and not expression.rhs.valid:
                    program += self.generate_multigrid(expression.rhs, storages, min_level, max_level,
                                                       use_global_weights)
                    expression.rhs.valid = True
                if not isinstance(expression.correction.approximation, system.Approximation):
                    program += self.generate_multigrid(expression.approximation, storages, min_level, max_level,
                                                       use_global_weights)
                if isinstance(expression.approximation, system.ZeroApproximation):
                    for i, grid in enumerate(expression.grid):
                        solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                        program += f'\t{solution_field.to_exa()} = 0\n'
                for i, grid in enumerate(expression.grid):
                    level = grid.level
                    solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                    rhs_field = self.get_rhs_field(storages, i, level)
                    operator = correction.operator
                    program += f'\t{solution_field.to_exa()} += {weight} * ({rhs_field.to_exa()}'
                    for j, entry in enumerate(operator.entries[i]):
                        field = self.get_solution_field(storages, j, grid.level, max_level)
                        if isinstance(entry, base.Identity):
                            program += field.to_exa()
                        elif isinstance(entry, base.ZeroOperator):
                            pass
                        else:
                            sympy_expr = self.generate_sympy_expression_for_operator_entry(entry, level)
                            sympy_expr = sympy_expr * sympy.MatrixSymbol(field.to_exa(), entry.shape[1], entry.shape[1])
                            program += f' - ({sympy_expr.expand()})'
                    program += ')\n'
            elif isinstance(correction, base.Multiplication):
                if isinstance(correction.operand1, system.InterGridOperator):
                    program += self.generate_multigrid(correction.operand2, storages, min_level, max_level,
                                                       use_global_weights)
                    for i, grid in enumerate(expression.grid):
                        solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                        operator = correction.operand1
                        entry = operator.entries[i][i]
                        if isinstance(entry, base.Prolongation):
                            op_level = entry.coarse_grid.level
                        elif isinstance(entry, base.Restriction):
                            op_level = entry.fine_grid.level
                        else:
                            raise RuntimeError("Unexpected entry")
                        source_field = self.obtain_correct_source_field(correction.operand2, storages, i, op_level, max_level)
                        program += f'\t{solution_field.to_exa()} += {weight} * ({entry.name}@{op_level} * ' \
                                   f'{source_field.to_exa()})\n'
                elif isinstance(correction.operand1, base.Inverse) or isinstance(correction.operand1, krylov_subspace.KrylovSubspaceMethod):
                    residual = correction.operand2
                    if not isinstance(residual.rhs, system.RightHandSide) and not residual.rhs.valid:
                        program += self.generate_multigrid(residual.rhs, storages, min_level, max_level,
                                                           use_global_weights)
                        residual.rhs.valid = True
                    if not isinstance(residual.approximation, system.Approximation):
                        program += self.generate_multigrid(residual.approximation, storages, min_level, max_level,
                                                           use_global_weights)
                    if isinstance(expression.approximation, system.ZeroApproximation):
                        for i, grid in enumerate(expression.grid):
                            solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                            program += f'\t{solution_field.to_exa()} = 0\n'

                    if isinstance(correction.operand1, krylov_subspace.KrylovSubspaceMethod):
                        level = expression.grid[0].level
                        krylov_subspace_operator = correction.operand1
                        program += f'\t{krylov_subspace_operator.name}@{level}()\n'
                        self.set_solver_valid(level, krylov_subspace_operator.name,
                                              krylov_subspace_operator.number_of_iterations)
                    elif isinstance(correction.operand1, base.Inverse):
                        smoothing_operator = correction.operand1.operand
                        system_operator = correction.operand2.operator
                        equation_dict = transformations.obtain_sympy_expression_for_local_system(smoothing_operator, system_operator,
                                                                                                 self.equations, self.fields)
                        dependent_equations, independent_equations = transformations.find_independent_equation_sets(equation_dict)
                        if isinstance(correction.operand1.operand, system.ElementwiseDiagonal):
                            dependent_equations.extend(independent_equations)
                            independent_equations.clear()
                        for key, value in independent_equations:
                            coloring = False
                            indentation = ''
                            if expression.partitioning == part.RedBlack:
                                coloring = True
                                program += '\tcolor with {\n\t\t(('
                                for i in range(self.dimension):
                                    program += f'i{i}'
                                    if i < self.dimension - 1:
                                        program += ' + '
                                program += ') % 2),\n'
                                indentation += '\t'
                            if key[1] < max_level:
                                program += f'\t{indentation}solve locally at gen_error_{key[0]}@{key[1]} relax {weight} {{\n'
                            else:
                                program += f'\t{indentation}solve locally at {key[0]}@{key[1]} relax {weight} {{\n'
                            program += self.generate_solve_locally(key, value, indentation, max_level)
                            program += f'\t{indentation}}}\n'
                            if coloring:
                                program += '\t}\n'

                        coloring = False
                        indentation = ''
                        if len(dependent_equations) > 0:
                            if expression.partitioning == part.RedBlack:
                                coloring = True
                                program += '\tcolor with {\n\t\t(('
                                for i in range(self.dimension):
                                    program += f'i{i}'
                                    if i < self.dimension - 1:
                                        program += ' + '
                                program += ') % 2),\n'
                                indentation += '\t'
                            level = dependent_equations[0][0][1]
                            if level < max_level:
                                program += f'\t{indentation}solve locally at gen_error_{dependent_equations[0][0][0]}@{dependent_equations[0][0][1]} relax {weight} {{\n'
                            else:
                                program += f'\t{indentation}solve locally at {dependent_equations[0][0][0]}@{dependent_equations[0][0][1]} relax {weight} {{\n'
                            for key, value in dependent_equations:
                                program += self.generate_solve_locally(key, value, indentation, max_level)
                            program += f'\t{indentation}}}\n'
                            if coloring:
                                program += '\t}\n'
                else:
                    raise RuntimeError("Unsupported operator")
            else:
                raise RuntimeError("Expected multiplication")
        elif isinstance(expression, base.Residual):
            if not isinstance(expression.rhs, system.RightHandSide) and not expression.rhs.valid:
                program += self.generate_multigrid(expression.rhs, storages, min_level, max_level, use_global_weights)
                expression.rhs.valid = True
            if not isinstance(expression.approximation, system.Approximation):
                program += self.generate_multigrid(expression.approximation, storages, min_level, max_level, use_global_weights)
            if isinstance(expression.approximation, system.ZeroApproximation):
                for i, grid in enumerate(expression.grid):
                    solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                    program += f'\t{solution_field.to_exa()} = 0\n'
            for i, grid in enumerate(expression.grid):
                level = grid.level
                residual_field = self.get_residual_field(storages, i, level)
                rhs_field = self.get_rhs_field(storages, i, level)
                operator = expression.operator
                program += f'\t{residual_field.to_exa()} = {rhs_field.to_exa()}'
                for j, entry in enumerate(operator.entries[i]):
                    field = self.get_solution_field(storages, j, grid.level, max_level)
                    if isinstance(entry, base.Identity):
                        program += field.to_exa()
                    elif isinstance(entry, base.ZeroOperator):
                        pass
                    else:
                        sympy_expr = self.generate_sympy_expression_for_operator_entry(entry, level)
                        sympy_expr = sympy_expr * sympy.MatrixSymbol(field.to_exa(), entry.shape[1], entry.shape[1])
                        program += f' - ({sympy_expr.expand()})'
                program += '\n'
        elif isinstance(expression, base.Multiplication):
            if isinstance(expression.operand1, system.InterGridOperator):
                program += self.generate_multigrid(expression.operand2, storages, min_level, max_level,
                                                   use_global_weights)
                for i, grid in enumerate(expression.grid):
                    operator = expression.operand1
                    entry = operator.entries[i][i]
                    if isinstance(entry, base.Prolongation):
                        op_level = entry.coarse_grid.level
                    elif isinstance(entry, base.Restriction):
                        op_level = entry.fine_grid.level
                    else:
                        raise RuntimeError("Unexpected entry")
                    source_field = self.obtain_correct_source_field(expression.operand2, storages, i, op_level, max_level)
                    if isinstance(operator, system.Restriction):
                        target_field = self.get_rhs_field(storages, i, grid.level)
                    else:
                        target_field = self.get_correction_field(storages, i, grid.level)
                    program += f'\t{target_field.to_exa()} = {entry.name}@{op_level} * ' \
                               f'{source_field.to_exa()}\n'
            elif isinstance(expression.operand1, base.CoarseGridSolver):
                program += self.generate_multigrid(expression.operand2, storages, min_level, max_level, use_global_weights)
                level = max_level
                for i, grid in enumerate(expression.operand2.grid):
                    # solution_field = self.get_solution_field(storages, i, grid.level)
                    # rhs_field = self.get_rhs_field(storages, i, grid.level)
                    source_field = self.get_rhs_field(storages, i, grid.level)
                    # program += f'\t{solution_field.to_exa()} = 0\n'
                    # tmp = rhs_field.to_exa()

                    # solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                    # program += f'\t{solution_field.to_exa()} = 0\n'
                    # program += f'\tgen_rhs_{self.fields[i]}@{grid.level} = {source_field.to_exa()}\n'
                    # program += f'\tgen_error_{self.fields[i]}@{grid.level} = 0\n'
                    # TODO fix
                    level = min(level, grid.level)
                    if grid.level == min_level:
                        program += f'\tgen_rhs_{self.fields[i]}@{grid.level} = {source_field.to_exa()}\n'
                        program += f'\tgen_error_{self.fields[i]}@{grid.level} = 0\n'
                    else:
                        solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                        program += f'\t{solution_field.to_exa()} = 0\n'
                program += f'\tgen_mgCycle@{level}()\n'
                for i, grid in enumerate(expression.grid):
                    target_field = self.get_correction_field(storages, i, grid.level)

                    solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                    program += f'\t{target_field.to_exa()} = {solution_field.to_exa()}\n'

                    # solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                    # program += f'\t{target_field.to_exa()} = {solution_field.to_exa()}\n'
                    # TODO fix
                    if grid.level == min_level:
                        pass
                        # program += f'\t{target_field.to_exa()} = gen_error_{self.fields[i]}@{grid.level}\n'
                    else:
                        solution_field = self.get_solution_field(storages, i, grid.level, max_level)
                        program += f'\t{target_field.to_exa()} = {solution_field.to_exa()}\n'
            else:
                raise RuntimeError("Not implemented")
        else:
            raise RuntimeError("Not implemented")
        return program

    def generate_l3_file(self, min_level, max_level, program: str):
        # TODO fix hacky solution
        input_file_path = f'{self._base_path_prefix}/{self.problem_name}_base_{self.mpi_rank}.exa3'
        output_file_path = \
            f'{self._base_path_prefix}/{self.problem_name}_{self.mpi_rank}.exa3'
        with open(f'{self.base_path}/{input_file_path}', 'r') as input_file:
            with open(f'{self.base_path}/{output_file_path}', 'w') as output_file:
                for field_declaration in self._field_declaration_cache:
                    output_file.write(field_declaration)
                line = input_file.readline()
                while line:
                    # TODO perform this check in a more accurate way
                    if ('Function gen_mgCycle@' in line and
                        any([f'Function gen_mgCycle@{level}' in line for level in range(min_level+1, max_level+1)]))\
                            or 'Function InitFields' in line:
                        while line and line[0] is not '}':
                            line = input_file.readline()
                        line = input_file.readline()
                    if 'Field' not in line or line not in self._field_declaration_cache:
                        output_file.write(line)
                    line = input_file.readline()
                    while line == '\n':
                        line = input_file.readline()

                output_file.write(program)

    def generate_adapted_settings_file(self, l2file_required=False):
        base_path = self.base_path
        input_file_path = self.settings_path
        output_file_path = self.settings_path_generated
        with open(f'{base_path}/{input_file_path}', 'r') as input_file:
            with open(f'{base_path}/{output_file_path}', 'w') as output_file:
                for line in input_file:
                    tokens = line.split('=')
                    lhs = tokens[0].strip(' \n\t')
                    if lhs == 'configName':
                        output_file.write(f'  {lhs}\t = "{self.problem_name}_{self.mpi_rank}"\n')
                    elif l2file_required:
                        output_file.write(line)
                    elif not lhs == 'l2file':
                        output_file.write(line)
        return output_file_path

        # subprocess.run(['cp', f'{self.base_path}/{relative_input_file_path}',
        #                 f'{self.base_path}/{relative_input_file_path}.backup'],
        #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # subprocess.run(
        #     ['cp', f'{self.base_path}/{relative_output_file_path}', f'{self.base_path}/{relative_input_file_path}'],
        #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def generate_level_adapted_knowledge_file(self, min_level: int, max_level: int):
        base_path = self.base_path
        input_file_path = self.knowledge_path
        # relative_output_file_path = f'{self.knowledge_path}.tmp'

        output_file_path = self.knowledge_path_generated
        with open(f'{base_path}/{input_file_path}', 'r') as input_file:
            with open(f'{base_path}/{output_file_path}', 'w') as output_file:
                for line in input_file:
                    tokens = line.split('=')
                    lhs = tokens[0].strip(' \n\t')
                    if lhs == 'minLevel':
                        output_file.write(f'  {lhs}\t= {min_level}\n')
                    elif lhs == 'maxLevel':
                        output_file.write(f'  {lhs}\t= {max_level}\n')
                    else:
                        output_file.write(line)
        return output_file_path

    def generate_adapted_layer_files(self, iteration_limit, coarse_grid_solver_type=None, number_of_cgs_iterations=None):
        base_path = self.base_path
        input_file_path = f'{self._base_path_prefix}/{self.problem_name}.exa3'
        output_file_path = f'{self._base_path_prefix}/{self.problem_name}_{self.mpi_rank}.exa3'
        tmp = f'{self.base_path}/{self._base_path_prefix}/{self.problem_name}'
        subprocess.run(['cp', f'{tmp}.exa1', f'{tmp}_{self.mpi_rank}.exa1'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout_copy_file)
        subprocess.run(['cp', f'{tmp}.exa2', f'{tmp}_{self.mpi_rank}.exa2'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout_copy_file)
        subprocess.run(['cp', f'{tmp}.exa4', f'{tmp}_{self.mpi_rank}.exa4'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout_copy_file)

        with open(f'{base_path}/{input_file_path}', 'r') as input_file:
            with open(f'{base_path}/{output_file_path}', 'w') as output_file:
                for line in input_file:
                    tokens = line.split('=')
                    lhs = tokens[0].strip(' \n\t')
                    if lhs == 'solver_maxNumIts':
                        output_file.write(f'  {lhs}\t= {iteration_limit}\n')
                    elif coarse_grid_solver_type is not None and lhs == 'solver_cgs':
                        output_file.write(f'  {lhs}\t= "{coarse_grid_solver_type}"\n')
                    elif number_of_cgs_iterations is not None and lhs == 'solver_cgs_maxNumIts':
                        output_file.write(f'  {lhs}\t= {number_of_cgs_iterations}\n')
                    else:
                        output_file.write(line)
        return output_file_path

    def extract_krylov_subspace_method_from_layer3_file(self, layer3_file_path, level):
        krylov_solver_function = ''
        residual_norm_function = ''
        with open(f'{self.base_path}/{layer3_file_path}', 'r') as input_file:
            line = input_file.readline()
            while line:
                if f'Function gen_mgCycle@{level}' in line:
                    block_count = 1
                    while line and block_count > 0:
                        if 'print' not in line:
                            krylov_solver_function += line
                        line = input_file.readline()
                        if '{' in line:
                            block_count += 1
                        elif '}' in line:
                            block_count -= 1
                    krylov_solver_function += line
                elif f'Function gen_resNorm@{level}' in line:
                    block_count = 1
                    while line and block_count > 0:
                        if 'print' not in line:
                            residual_norm_function += line
                        line = input_file.readline()
                        if '{' in line:
                            block_count += 1
                        elif '}' in line:
                            block_count -= 1
                    residual_norm_function += line
                elif 'Field' in line and 'gen_' in line:
                    self._field_declaration_cache.add(line)
                line = input_file.readline()
        return krylov_solver_function, residual_norm_function

    def add_solver_to_cache(self, level, solver_type: str, number_of_solver_iterations: int, program: str, valid=False):
        key = level, solver_type, number_of_solver_iterations
        self._solver_cache[key] = program, valid

    def solver_in_cache(self, level, solver_type: str, number_of_solver_iterations: int):
        key = level, solver_type, number_of_solver_iterations
        return key in self._solver_cache

    def set_solver_valid(self, level, solver_type: str, number_of_solver_iterations: int):
        key = level, solver_type, number_of_solver_iterations
        value = self._solver_cache[key]
        self._solver_cache[key] = value[0], True

    def invalidate_solver_cache(self):
        self._solver_cache = {key: (value[0], False) for key, value in self._solver_cache.items()}

    def generate_krylov_subspace_method(self, level: int, max_level: int, solver_type: str,
                                        number_of_solver_iterations: int):
        min_level = level
        iteration_limit = 1
        knowledge_path = self.generate_level_adapted_knowledge_file(min_level, min(min_level + 1, max_level))
        self.generate_adapted_layer_files(iteration_limit, solver_type, number_of_solver_iterations)
        settings_path = self.generate_adapted_settings_file(l2file_required=True)
        self.run_exastencils_compiler(knowledge_path=knowledge_path, settings_path=settings_path)
        layer3_file_path = f'{self._debug_l3_path}'.replace('_debug.exa3', f'_{self.mpi_rank}_debug.exa3')
        krylov_solver_function, residual_norm_function = \
            self.extract_krylov_subspace_method_from_layer3_file(layer3_file_path, level)
        krylov_solver_function = krylov_solver_function.replace('gen_mgCycle', solver_type)
        return krylov_solver_function, residual_norm_function

    def generate_cached_krylov_subspace_solvers(self, min_level, max_level, solver_list, maximum_number_of_solver_iterations):
        self._field_declaration_cache.clear()
        residual_norm_functions = []
        for level in range(min_level, max_level + 1):
            for i, solver_type in enumerate(solver_list):
                for number_of_solver_iterations in range(1, maximum_number_of_solver_iterations+1):
                    krylov_solver_function, residual_norm_function = \
                        self.generate_krylov_subspace_method(level, max_level, solver_type, number_of_solver_iterations)
                    self.add_solver_to_cache(level, solver_type, number_of_solver_iterations, krylov_solver_function)
                    if i == 0:
                        residual_norm_functions.append(residual_norm_function)
        return residual_norm_functions





