from evostencils.expressions import base, multigrid as mg
from evostencils.stencils import constant

class CycleStorage:
    def __init__(self, level, grid):
        self.level = level
        self.grid = grid
        self.solution = Field(f'Solution', level, self)
        self.rhs = Field(f'RHS', level, self)
        self.residual = Field(f'Residual', level, self)
        self.correction = Field(f'Correction', level, self)


class Field:
    def __init__(self, name=None, level=None, cycle_storage=None):
        self.name = name
        self.level = level
        self.cycle_storage = cycle_storage


class ProgramGenerator:
    def __init__(self, op: base.Operator, grid: base.Grid, rhs: base.Grid, dimension, coarsening_factor,
                 interpolation_stencil_generator=None, restriction_stencil_generator=None):
        self._operator = op
        self._grid = grid
        self._rhs = rhs
        self._dimension= dimension
        self._coarsening_factor = coarsening_factor
        self._interpolation_stencil_generator = interpolation_stencil_generator
        self._restriction_stencil_generator = restriction_stencil_generator

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
    def interpolation_stencil_generator(self):
        return self._interpolation_stencil_generator

    @property
    def restriction_stencil_generator(self):
        return self._restriction_stencil_generator

    def generate(self, expression: mg.Cycle):
        max_level = self.obtain_maximum_level(expression)
        storages = self.generate_storage(max_level)
        expression.storage = storages[0].solution
        self.assign_storage_to_cycles(expression, storages, 0)
        program = str('')
        program = self.add_field_declarations_to_program_string(program, storages)
        program = self.add_operator_declarations_to_program_string(program, max_level)
        return program

    @staticmethod
    def obtain_maximum_level(cycle: mg.Cycle) -> int:
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
        return recursive_descent(cycle, cycle.grid.size, 0) + 1

    def generate_storage(self, maximum_level):
        tmps = []
        grid = self.grid
        for level in range(maximum_level, -1, -1):
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
    def assign_storage_to_cycles(node: base.Expression, storages: [CycleStorage], i: int):
        if isinstance(node, mg.Cycle):
            i = ProgramGenerator.adjust_storage_index(node, storages, i)
            node.iterate.storage = storages[i].solution
            #node.rhs.storage = storages[i].rhs
            node.correction.storage = storages[i].correction
            ProgramGenerator.assign_storage_to_cycles(node.iterate, storages, i)
            ProgramGenerator.assign_storage_to_cycles(node.correction, storages, i)
        elif isinstance(node, mg.Residual):
            i = ProgramGenerator.adjust_storage_index(node, storages, i)
            node.iterate.storage = storages[i].solution
            node.rhs.storage = storages[i].rhs
            ProgramGenerator.assign_storage_to_cycles(node.iterate, storages, i)
            ProgramGenerator.assign_storage_to_cycles(node.rhs, storages, i)
        elif isinstance(node, base.BinaryExpression):
            operand1 = node.operand1
            operand2 = node.operand2
            if ProgramGenerator.needs_storage(operand2):
                i = ProgramGenerator.adjust_storage_index(operand2, storages, i)
                if isinstance(operand2, mg.Residual):
                    operand2.storage = storages[i].residual
                else:
                    operand2.storage = storages[i].correction
            ProgramGenerator.assign_storage_to_cycles(operand1, storages, i)
            ProgramGenerator.assign_storage_to_cycles(operand2, storages, i)
        elif isinstance(node, base.UnaryExpression) or isinstance(node, base.Scaling):
            operand = node.operand
            if ProgramGenerator.needs_storage(operand):
                i = ProgramGenerator.adjust_storage_index(operand, storages, i)
                operand.storage = storages[i].correction

    @staticmethod
    def add_field_declarations_to_program_string(program_string: str, storages: [CycleStorage]):
        for storage in storages:
            program_string += f'Field {storage.solution.name}@{storage.solution.level} with Real on Node of global = 0.0\n'
            program_string += f'Field {storage.rhs.name}@{storage.rhs.level} with Real on Node of global = 0.0\n'
            program_string += f'Field {storage.residual.name}@{storage.residual.level} with Real on Node of global = 0.0\n'
            program_string += f'Field {storage.correction.name}@{storage.residual.level} with Real on Node of global = 0.0\n'
        return program_string

    @staticmethod
    def add_constant_stencil_to_program_string(program_string: str, level: int, operator_name: str, stencil: constant.Stencil):
        program_string += f"Operator {operator_name}@{level} from Stencil {{\n"
        for offset, value in stencil.entries:
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

    def add_operator_declarations_to_program_string(self, program_string: str, maximum_level: int):
        grid = self.grid
        for i in range(maximum_level + 1):
            stencil = self.operator.stencil_generator(grid)
            grid = mg.get_coarse_grid(grid, self.coarsening_factor)
            program_string = self.add_constant_stencil_to_program_string(program_string, i, self.operator.name, stencil)

        if self._interpolation_stencil_generator is None:
            program_string += "Operator Prolongation from default prolongation on Node with 'linear'"
        else:
            grid = self.grid
            for i in range(maximum_level):
                stencil = self.interpolation_stencil_generator(grid)
                # TODO generate string for stencil
                grid = mg.get_coarse_grid(grid, self.coarsening_factor)

        if self._restriction_stencil_generator is None:
            program_string += "Operator Restriction from default restriction on Node with 'linear'"
        else:
            grid = self.grid
            for i in range(maximum_level):
                stencil = self.restriction_stencil_generator(grid)
                # TODO generate string for stencil
                grid = mg.get_coarse_grid(grid, self.coarsening_factor)

        return program_string
