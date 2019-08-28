from evostencils.expressions import base, partitioning as part, system
from evostencils.stencils import constant, periodic
from evostencils.initialization import multigrid
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
    def __init__(self, problem_name: str, exastencils_path: str, dimension, finest_grid, coarsening_factor, min_level, max_level, equations,
                 operators, fields, output_path="./execution"):
        self.problem_name = problem_name
        self._exastencils_path = exastencils_path
        self._output_path = output_path
        self._dimension = dimension
        self._finest_grid = finest_grid
        self._coarsening_factor = coarsening_factor
        self._min_level = min_level
        self._max_level = max_level
        self._equations = equations
        self._operators = operators
        self._fields = fields
        self._compiler_available = False
        self._performance_estimate_file_name = 'performance_estimate.csv'
        self._compiler_file_name = 'compiler.jar'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if os.path.exists(exastencils_path):
            subprocess.run(['cp', '-r', f'{exastencils_path}/Examples/lib', f'{output_path}/'])
            if os.path.isfile(f'{exastencils_path}/Compiler/{self.compiler_file_name}'):
                self._compiler_available = True
        # Settings and knowledge for execution
        # Settings and knowledge for performance estimation
        tmp = f'{problem_name}_performance_estimation'

    @property
    def compiler_file_name(self):
        return self._compiler_file_name

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

    def run_exastencils_compiler(self, input_name=None, platform='linux'):
        if input_name is None:
            input_name = self.problem_name
        result = subprocess.run(['java', '-cp',
                                 f'{self.exastencils_path}/Compiler/{self.compiler_file_name}', 'Main',
                                 f'{self.output_path}/{input_name}.settings',
                                 f'{self.output_path}/{input_name}.knowledge',
                                 f'{self.output_path}/lib/{platform}.platform'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.returncode

    def estimate_runtime_per_iteration(self, platform='linux'):
        self.run_exastencils_compiler(f'{self.problem_name}_performance_estimation', platform)
        results = []
        with open(f'{self.output_path}/{self._performance_estimate_file_name}', 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                if 'cycle' in row[1].lower():
                    results.append(float(row[2]))
        estimated_runtime = max(results)
        return estimated_runtime

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

