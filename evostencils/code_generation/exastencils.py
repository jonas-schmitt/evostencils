from evostencils.expressions import base, partitioning as part, system
from evostencils.stencils import constant, periodic
from evostencils.initialization import multigrid, parser
import os
import subprocess
import math
import csv
import sympy


class CycleStorage:
    def __init__(self, level, equations: [multigrid.EquationInfo], fields: [sympy.Symbol], grid):
        self.level = level
        self.grid = grid
        self.solution = [Field(f'{symbol.name}', level, self) for symbol in fields]
        self.rhs = [Field(f'{eq_info.rhs_name}', level, self) for eq_info in equations]
        self.residual = [Field(f'Residual_{symbol.name}', level, self) for symbol in fields]
        self.correction = [Field(f'Correction_{symbol.name}', level, self) for symbol in fields]


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
    def __init__(self, compiler_path: str, base_path: str, relative_settings_path: str, relative_knowledge_path: str,
                 platform='linux'):
        self._compiler_path = compiler_path
        self._base_path = base_path
        self._relative_knowledge_path = relative_knowledge_path
        self._relative_settings_path = relative_settings_path
        parser.extract_settings_information(base_path, relative_settings_path)
        self._dimension, self._min_level, self._max_level = parser.extract_knowledge_information(base_path, relative_knowledge_path)
        self._base_path_prefix, self._config_name, self._debug_l3_file = parser.extract_settings_information(base_path, relative_settings_path)
        self._platform = platform
        self.run_exastencils_compiler(platform)
        self._equations, self._operators, self._fields = parser.extract_l2_information(f'{base_path}/{self._base_path_prefix}/{self._debug_l3_file}', self.dimension)
        size = 2 ** self._max_level
        grid_size = tuple([size] * self.dimension)
        h = 1 / (2 ** self._max_level)
        step_size = tuple([h] * self.dimension)
        tmp = tuple([2] * self.dimension)
        self._coarsening_factor = [tmp for _ in range(len(self.fields))]
        self._finest_grid = [base.Grid(grid_size, step_size) for _ in range(len(self.fields))]
        self._compiler_available = False
        self._compiler_path = compiler_path
        if os.path.exists(compiler_path):
            if os.path.isfile(compiler_path):
                self._compiler_available = True

    @property
    def compiler_path(self):
        return self._compiler_path

    @property
    def relative_knowledge_path(self):
        return self._relative_knowledge_path

    @property
    def relative_settings_path(self):
        return self._relative_settings_path

    @property
    def compiler_available(self):
        return self._compiler_available

    @property
    def base_path(self):
        return self._base_path

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
        path_to_file = f'{self.output_path}/generated/{self.problem_name}/Globals/Globals_initGlobals.cpp'
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
        path_to_file = f'{self.output_path}/generated/{self.problem_name}/Globals/Globals_initGlobals.cpp'
        subprocess.run(['cp', f'{path_to_file}.backup', path_to_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def generate_cycle_function(self, expression, storages, use_global_weights=False):
        base_level = 0
        for i, storage in enumerate(storages):
            if expression.grid.size == storage.grid.size:
                expression.storage = storage.solution
                base_level = i
                break
        self.assign_storage_to_subexpressions(expression, storages, base_level)
        program = f'Function Cycle@(finest - {base_level}) {{\n'
        program += self.generate_multigrid(expression, storages, use_global_weights)
        program += '}\n'
        return program

    def write_program_to_file(self, program: str):
        with open(f'{self.output_path}/{self.problem_name}.exa3', "w") as file:
            print(program, file=file)

    def run_exastencils_compiler(self, platform='linux'):
        result = subprocess.run(['java', '-cp',
                                 self.compiler_path, 'Main',
                                 f'{self.base_path}/{self.relative_settings_path}',
                                 f'{self.base_path}/{self.relative_knowledge_path}',
                                 f'{self.base_path}/lib/{platform}.platform'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.returncode

    def run_c_compiler(self):
        result = subprocess.run(['make', '-j4', '-s', '-C', f'{self.output_path}/generated/{self.problem_name}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.returncode

    def evaluate(self, platform='linux', infinity=1e100, number_of_samples=1, only_weights_adapted=False):
        if not only_weights_adapted:
            return_code = self.run_exastencils_compiler(self.problem_name, platform)
            if not return_code == 0:
                return infinity, infinity
        return_code = self.run_c_compiler()
        if not return_code == 0:
            return infinity, infinity
        total_time = 0
        sum_of_convergence_factors = 0
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

    def generate_storage(self, maximum_level):
        pass

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
        pass

    def generate_multigrid(self, expression: base.Expression, storages, use_global_weights=False):
        # import decimal
        # if expression.program is not None:
        #     return expression.program
        program = ''
        if expression.storage is not None:
            expression.storage.valid = False

    @staticmethod
    def invalidate_storages(storages: [CycleStorage]):
        for storage in storages:
            for residual, rhs, solution, correction in zip(storage.residual, storage.rhs, storage.solution, storage.correction):
                residual.valid = False
                rhs.valid = False
                solution.valid = False
                correction.valid = False

