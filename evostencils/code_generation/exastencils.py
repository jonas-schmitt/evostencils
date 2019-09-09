from evostencils.expressions import base, partitioning as part, system, transformations
from evostencils.initialization import multigrid, parser
import os
import subprocess
import math
import sympy


class CycleStorage:
    def __init__(self, equations: [multigrid.EquationInfo], fields: [sympy.Symbol], grid):
        self.grid = grid
        self.solution = [Field(f'{symbol.name}', g.level, self) for g, symbol in zip(grid, fields)]
        self.rhs = [Field(f'{eq_info.rhs_name}', g.level, self) for g, eq_info in zip(grid, equations)]
        self.residual = [Field(f'gen_residual_{symbol.name}', g.level, self) for g, symbol in zip(grid, fields)]
        self.correction = [Field(f'gen_error_{symbol.name}', g.level, self) for g, symbol in zip(grid, fields)]


class Field:
    def __init__(self, name=None, level=None, cycle_storage=None):
        self.name = name
        self.level = level
        self.cycle_storage = cycle_storage
        self.valid = False

    def to_exa(self):
        return f'{self.name}@{self.level}'


class ProgramGenerator:
    def __init__(self, absolute_compiler_path: str, base_path: str, settings_path: str, knowledge_path: str,
                 platform='linux'):
        self._absolute_compiler_path = absolute_compiler_path
        self._base_path = base_path
        self._knowledge_path = knowledge_path
        self._settings_path = settings_path
        self._dimension, self._min_level, self._max_level = \
            parser.extract_knowledge_information(base_path, knowledge_path)
        self._base_path_prefix, self._problem_name, self._debug_l3_path, self._output_path = \
            parser.extract_settings_information(base_path, settings_path)
        self._platform = platform
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

    def generate_global_weight_initializations(self, weights):
        # Hack to change the weights after generation
        weights = reversed(weights)
        path_to_file = f'{self.base_path}/{self.output_path}/Globals/Globals_initGlobals.cpp'
        subprocess.run(['cp', path_to_file, f'{path_to_file}.backup'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    def restore_global_initializations(self):
        # Hack to change the weights after generation
        path_to_file = f'{self.base_path}/{self.output_path}/Globals/Globals_initGlobals.cpp'
        subprocess.run(['cp', f'{path_to_file}.backup', path_to_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def get_solution_field(storages, index, level):
        max_level = storages[0].grid[index].level
        offset = max_level - level
        return storages[offset].solution[index]

    @staticmethod
    def get_rhs_field(storages, index, level):
        max_level = storages[0].grid[index].level
        offset = max_level - level
        return storages[offset].rhs[index]

    @staticmethod
    def get_residual_field(storages, index, level):
        max_level = storages[0].grid[index].level
        offset = max_level - level
        return storages[offset].residual[index]

    @staticmethod
    def get_correction_field(storages, index, level):
        max_level = storages[0].grid[index].level
        offset = max_level - level
        return storages[offset].correction[index]

    def generate_cycle_function(self, expression, storages, min_level, max_level, use_global_weights=False):
        program = f'Function gen_mgCycle@{max_level} {{\n'
        program += self.generate_multigrid(expression, storages, min_level, max_level, use_global_weights)
        program += '}\n'
        return program

    def run_exastencils_compiler(self):

        current_path = os.getcwd()
        os.chdir(self.base_path)
        result = subprocess.run(['java', '-cp',
                                 self.absolute_compiler_path, 'Main',
                                 f'{self.base_path}/{self.settings_path}',
                                 f'{self.base_path}/{self.knowledge_path}',
                                 f'{self.base_path}/lib/{self.platform}.platform'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        os.chdir(current_path)
        if result.returncode != 0:
            raise RuntimeError("Compiler not working. Aborting.")
        return result.returncode

    def run_c_compiler(self):
        result = subprocess.run(['make', '-j4', '-s', '-C', f'{self.base_path}/{self.output_path}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.returncode

    def evaluate(self, infinity=1e100, number_of_samples=1, only_weights_adapted=False):
        if not only_weights_adapted:
            return_code = self.run_exastencils_compiler()
            if not return_code == 0:
                return infinity, infinity
        return_code = self.run_c_compiler()
        if not return_code == 0:
            return infinity, infinity
        total_time = 0
        sum_of_convergence_factors = 0
        for i in range(number_of_samples):
            result = subprocess.run([f'{self.base_path}/{self.output_path}/exastencils'],
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

    def generate_storage(self, min_level, max_level, finest_grid):
        storage = []
        grid = finest_grid
        for i in range(min_level, max_level):
            storage.append(CycleStorage(self.equations, self.fields, grid))
            grid = system.get_coarse_grid(grid, self.coarsening_factor)
        return storage

    @staticmethod
    def needs_storage(expression: base.Expression):
        return expression.shape[1] == 1

    # Warning: This function modifies the expression passed to it
    @staticmethod
    def assign_storage_to_subexpressions(expression: base.Expression, storages: [CycleStorage], i: int):
        if expression.storage is not None:
            return None

    def obtain_correct_source_field(self, expression, storages, i, level):
        if isinstance(expression, system.Approximation) or isinstance(expression, base.Cycle):
            return self.get_solution_field(storages, i, level)
        elif isinstance(expression, base.Residual):
            return self.get_residual_field(storages, i, level)
        elif isinstance(expression, system.RightHandSide):
            return self.get_rhs_field(storages, i, level)
        else:
            return self.get_correction_field(storages, i, level)

    @staticmethod
    def generate_solve_locally(key, value, indentation):
        program = ''
        rhs = f'{value[1]}@['
        unknown = f'{key[0]}@{key[1]}@['
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
        program += f'\t\t{indentation}{unknown} => ({transformed_equation}) == {rhs}\n'
        return program

    def generate_multigrid(self, expression: base.Expression, storages, min_level, max_level, use_global_weights=False):
        # import decimal
        # if expression.program is not None:
        #     return expression.program
        program = ''
        if isinstance(expression, base.Cycle):
            weight = expression.weight
            # Hack to change the weights after generation

            if use_global_weights and hasattr(expression, 'global_id'):
                weight = f'omega_{expression.global_id}'

            correction = expression.correction
            if isinstance(expression.approximation, system.ZeroApproximation):
                for i, grid in enumerate(expression.grid):
                    solution_field = self.get_solution_field(storages, i, grid.level)
                    program += f'\t{solution_field.to_exa()} = 0\n'
            if isinstance(correction, base.Residual):
                if not isinstance(expression.correction.approximation, system.Approximation):
                    program += self.generate_multigrid(expression.approximation, storages, min_level, max_level,
                                                       use_global_weights)
                if not isinstance(expression.correction.rhs, system.RightHandSide):
                    program += self.generate_multigrid(expression.rhs, storages, min_level, max_level,
                                                       use_global_weights)
                for i, grid in enumerate(expression.grid):
                    level = grid.level
                    solution_field = self.get_solution_field(storages, i, level)
                    rhs_field = self.get_rhs_field(storages, i, level)
                    operator = correction.operator
                    program += f'\t{solution_field.to_exa()} += {weight} * ({rhs_field.to_exa()}'
                    for j, entry in enumerate(operator.entries[i]):
                        field = self.get_solution_field(storages, j, level)
                        if isinstance(entry, base.Scaling) and not isinstance(entry.operand, base.ZeroOperator):
                            program += f' - '
                            op = entry.operand
                            program += f'({entry.factor}) * '
                        elif isinstance(entry, base.Operator):
                            op = entry
                            if not isinstance(op, base.ZeroOperator):
                                program += f' - '
                        else:
                            raise RuntimeError("Unexpected system operator entry")
                        if isinstance(op, base.Identity):
                            program += field.to_exa()
                        elif isinstance(op, base.ZeroOperator):
                            pass
                        else:
                            program += f'\t{op}@{level} * {field.to_exa()}'
                    program += ')\n'
            elif isinstance(correction, base.Multiplication):
                if isinstance(correction.operand1, system.InterGridOperator):
                    program += self.generate_multigrid(correction.operand2, storages, min_level, max_level,
                                                       use_global_weights)
                    for i, grid in enumerate(expression.grid):
                        level = grid.level
                        solution_field = self.get_solution_field(storages, i, level)
                        operator = correction.operand1
                        entry = operator.entries[i][i]
                        if isinstance(entry, base.Prolongation):
                            op_level = entry.coarse_grid.level
                        elif isinstance(entry, base.Restriction):
                            op_level = entry.fine_grid.level
                        else:
                            raise RuntimeError("Unexpected entry")
                        source_field = self.obtain_correct_source_field(correction.operand2, storages, i, op_level)
                        program += f'\t{solution_field.to_exa()} += {weight} * ({entry.name}@{op_level} * ' \
                                   f'{source_field.to_exa()})\n'
                elif isinstance(correction.operand1, base.Inverse):
                    residual = correction.operand2
                    if not isinstance(residual.approximation, system.Approximation):
                        program += self.generate_multigrid(residual.approximation, storages, min_level, max_level,
                                                           use_global_weights)
                    if not isinstance(residual.rhs, system.RightHandSide):
                        program += self.generate_multigrid(residual.rhs, storages, min_level, max_level,
                                                           use_global_weights)
                    smoothing_operator = correction.operand1.operand
                    system_operator = correction.operand2.operator
                    equation_dict = transformations.obtain_sympy_expression_for_local_system(smoothing_operator, system_operator,
                                                                                             self.equations, self.fields)
                    dependent_equations, independent_equations = transformations.find_independent_equation_sets(equation_dict)
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
                        program += f'\t{indentation}solve locally at {key[0]}@{key[1]} relax {weight} {{\n'
                        program += self.generate_solve_locally(key, value, indentation)
                        program += f'\t{indentation}}}\n'
                        if coloring:
                            program += '\t\t}\n\t}\n'

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
                    if len(dependent_equations) > 0:
                        program += f'\t{indentation}solve locally at {dependent_equations[0][0][0]}@{dependent_equations[0][0][1]} relax {weight} {{\n'
                        for key, value in dependent_equations:
                            program += self.generate_solve_locally(key, value, indentation)
                        program += f'\t{indentation}}}\n'
                    if coloring:
                        program += '\t\t}\n\t}\n'
                else:
                    raise RuntimeError("Unsupported operator")
            else:
                raise RuntimeError("Expected multiplication")
        elif isinstance(expression, base.Residual):
            if not isinstance(expression.approximation, system.Approximation):
                program += self.generate_multigrid(expression.approximation, storages, min_level, max_level, use_global_weights)
            if not isinstance(expression.rhs, system.RightHandSide):
                program += self.generate_multigrid(expression.rhs, storages, min_level, max_level, use_global_weights)
            for i, grid in enumerate(expression.grid):
                level = grid.level
                residual_field = self.get_residual_field(storages, i, level)
                rhs_field = self.get_rhs_field(storages, i, level)
                operator = expression.operator
                program += f'\t{residual_field.to_exa()} = {rhs_field.to_exa()}'
                for j, entry in enumerate(operator.entries[i]):
                    field = self.get_solution_field(storages, j, level)
                    if isinstance(entry, base.Scaling) and not isinstance(entry.operand, base.ZeroOperator):
                        program += f' - '
                        op = entry.operand
                        program += f'({entry.factor}) * '
                    elif isinstance(entry, base.Operator):
                        op = entry
                        if not isinstance(op, base.ZeroOperator):
                            program += f' - '
                    else:
                        raise RuntimeError("Unexpected system operator entry")
                    if isinstance(op, base.Identity):
                        program += field.to_exa()
                    elif isinstance(op, base.ZeroOperator):
                        pass
                    else:
                        program += f'{op}@{level} * {field.to_exa()}'
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
                    source_field = self.obtain_correct_source_field(expression.operand2, storages, i, op_level)
                    target_field = self.get_correction_field(storages, i, grid.level)
                    program += f'\t{target_field.to_exa()} = {entry.name}@{op_level} * ' \
                               f'{source_field.to_exa()}\n'
            elif isinstance(expression.operand1, base.CoarseGridSolver):
                program += self.generate_multigrid(expression.operand2, storages, min_level, max_level, use_global_weights)
                for i, grid in enumerate(expression.operand2.grid):
                    solution_field = self.get_solution_field(storages, i, grid.level)
                    rhs_field = self.get_rhs_field(storages, i, grid.level)
                    source_field = self.obtain_correct_source_field(expression.operand2, storages, i, grid.level)
                    program += f'\t{solution_field.to_exa()} = 0\n'
                    tmp = rhs_field.to_exa()
                    program += f'\t{tmp} = {source_field.to_exa()}\n'
                program += f'\tgen_mgCycle@{min_level}()\n'
                for i, grid in enumerate(expression.grid):
                    source_field = self.get_solution_field(storages, i, grid.level)
                    target_field = self.get_correction_field(storages, i, grid.level)
                    program += f'\t{target_field.to_exa()} = {source_field.to_exa()}\n'
            else:
                raise RuntimeError("Not implemented")
        else:
            raise RuntimeError("Not implemented")
        return program

    @staticmethod
    def invalidate_storages(storages: [CycleStorage]):
        for storage in storages:
            for residual, rhs, solution, correction in zip(storage.residual, storage.rhs, storage.solution, storage.correction):
                residual.valid = False
                rhs.valid = False
                solution.valid = False
                correction.valid = False

