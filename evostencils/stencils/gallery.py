import abc
from evostencils.stencils import constant


class StencilGenerator(abc.ABC):

    @abc.abstractmethod
    def generate_stencil(self, grid):
        pass

    @abc.abstractmethod
    def generate_exa3(self, name):
        pass


class Poisson1D(StencilGenerator):

    def generate_stencil(self, grid):
        h, = grid.spacing
        entries = [
            ((-1,), -1 / (h * h)),
            ((0,), 2 / (h * h)),
            ((1,), -1 / (h * h)),
        ]
        return constant.Stencil(entries)

    def generate_exa3(self, name):
        # TODO implement this function
        pass


class Poisson2D(StencilGenerator):

    def generate_stencil(self, grid):
        eps = 1.0
        h0, h1 = grid.spacing
        entries = [
            ((0, -1), -1 / (h1 * h1)),
            ((-1, 0), -1 / (h0 * h0) * eps),
            ((0, 0), 2 / (h0 * h0) * eps + 2 / (h1 * h1)),
            ((1, 0), -1 / (h0 * h0) * eps),
            ((0, 1), -1 / (h1 * h1))
        ]
        return constant.Stencil(entries)

    def generate_exa3(self, name):
        return """
Operator A from Stencil {
    [ 0,  0] =>  2.0 / ( vf_gridWidth_x ** 2 ) + 2.0 / ( vf_gridWidth_y ** 2 )
    [-1,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 1,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 0, -1] => -1.0 / ( vf_gridWidth_y ** 2 )
    [ 0,  1] => -1.0 / ( vf_gridWidth_y ** 2 )
}
"""


class Poisson3D(StencilGenerator):

    def generate_stencil(self, grid):
        h0, h1, h2 = grid.spacing
        entries = [
            ((0, 0, 0), 2 / (h0 * h0) + 2 / (h1 * h1) + 2 / (h2 * h2)),
            ((-1, 0, 0), -1 / (h0 * h0)),
            ((1, 0, 0), -1 / (h0 * h0)),
            ((0, -1, 0), -1 / (h1 * h1)),
            ((0, 1, 0), -1 / (h1 * h1)),
            ((0, 0, -1), -1 / (h2 * h2)),
            ((0, 0, 1), -1 / (h2 * h2))
        ]
        return constant.Stencil(entries)

    def generate_exa3(self, name):
        return """
Operator A from Stencil {
    [ 0,  0,  0] =>  2.0 / ( vf_gridWidth_x ** 2 ) + 2.0 / ( vf_gridWidth_y ** 2 ) + 2.0 / ( vf_gridWidth_z ** 2 )
    [-1,  0,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 1,  0,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 0, -1,  0] => -1.0 / ( vf_gridWidth_y ** 2 )
    [ 0,  1,  0] => -1.0 / ( vf_gridWidth_y ** 2 )
    [ 0,  0, -1] => -1.0 / ( vf_gridWidth_z ** 2 )
    [ 0,  0,  1] => -1.0 / ( vf_gridWidth_z ** 2 )
}
"""


def get_coefficient_2D(pos_x, pos_y):
    from math import exp
    kappa = 10.0
    return exp((kappa * ((pos_x - (pos_x * pos_x)) * (pos_y - (pos_y * pos_y)))))


class Poisson2DVariableCoefficients(StencilGenerator):
    def __init__(self, coefficient_function, position):
        assert len(position) == 2, 'Position must be a two dimensional array'
        self.get_coefficient = coefficient_function
        self.position = position

    def generate_stencil(self, grid):
        pos_x = self.position[0]
        pos_y = self.position[1]
        width_x, width_y = grid.spacing
        entries = [
            ((0, 0), (((self.get_coefficient((pos_x + (0.5 * width_x)), pos_y) + self.get_coefficient((pos_x - (0.5 * width_x)), pos_y))
                       / (width_x * width_x))
                      + ((self.get_coefficient(pos_x, (pos_y + (0.5 * width_y))) + self.get_coefficient(pos_x, (pos_y - (0.5 * width_y))))
                         / (width_y * width_y)))),
            ((1, 0), ((-1.0 * self.get_coefficient((pos_x + (0.5 * width_x)), pos_y))
                      / (width_x * width_x))),
            ((-1, 0), ((-1.0 * self.get_coefficient((pos_x - (0.5 * width_x)), pos_y))
                       / (width_x * width_x))),
            ((0, 1),  ((-1.0 * self.get_coefficient(pos_x, (pos_y + (0.5 * width_y))))
                       / (width_y * width_y))),
            ((0, -1), ((-1.0 * self.get_coefficient(pos_x, (pos_y - (0.5 * width_y))))
                       / (width_y * width_y)))
        ]
        return constant.Stencil(entries)

    def generate_exa3(self, name):
        return """
Globals {
    Val kappa : Real = 10.0
}

Function getCoefficient ( xPos : Real, yPos : Real ) : Real {
    return exp ( ( kappa * ( ( xPos - ( xPos ** 2 ) ) * ( yPos - ( yPos ** 2 ) ) ) ) )
}

Operator A from Stencil {
    [0, 0] => ( ( ( getCoefficient ( ( vf_nodePosition_x@current + ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) + getCoefficient ( ( vf_nodePosition_x@current - ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) ) + ( ( getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current + ( 0.5 * vf_gridWidth_y@current ) ) ) + getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current - ( 0.5 * vf_gridWidth_y@current ) ) ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) ) )
    [1, 0] => ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x@current + ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) )
    [-1, 0] => ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x@current - ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) )
    [0, 1] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current + ( 0.5 * vf_gridWidth_y@current ) ) ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) )
    [0, -1] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current - ( 0.5 * vf_gridWidth_y@current ) ) ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) )
}
"""


def get_coefficient_3D(pos_x, pos_y, pos_z):
    from math import exp
    kappa = 10.0
    return exp ((kappa * (((pos_x - (pos_x ** 2)) * (pos_y - (pos_y ** 2))) * (pos_z - (pos_z ** 2)))))


class Poisson3DVariableCoefficients(StencilGenerator):
    def __init__(self, coefficient_function, position):
        assert len(position) == 3, 'Position must be a three dimensional array'
        self.get_coefficient = coefficient_function
        self.position = position

    def generate_stencil(self, grid):
        vf_nodePosition_x, vf_nodePosition_y, vf_nodePosition_z = self.position
        vf_gridWidth_x, vf_gridWidth_y, vf_gridWidth_z = grid.spacing
        getCoefficient = self.get_coefficient
        entries = [
            ((0, 0, 0), ( ( ( ( getCoefficient ( ( vf_nodePosition_x + ( 0.5 * vf_gridWidth_x ) ), vf_nodePosition_y, vf_nodePosition_z ) + getCoefficient ( ( vf_nodePosition_x - ( 0.5 * vf_gridWidth_x ) ), vf_nodePosition_y, vf_nodePosition_z ) ) / ( vf_gridWidth_x * vf_gridWidth_x ) ) + ( ( getCoefficient ( vf_nodePosition_x, ( vf_nodePosition_y + ( 0.5 * vf_gridWidth_y ) ), vf_nodePosition_z ) + getCoefficient ( vf_nodePosition_x, ( vf_nodePosition_y - ( 0.5 * vf_gridWidth_y ) ), vf_nodePosition_z ) ) / ( vf_gridWidth_y * vf_gridWidth_y ) ) ) + ( ( getCoefficient ( vf_nodePosition_x, vf_nodePosition_y, ( vf_nodePosition_z + ( 0.5 * vf_gridWidth_z ) ) ) + getCoefficient ( vf_nodePosition_x, vf_nodePosition_y, ( vf_nodePosition_z - ( 0.5 * vf_gridWidth_z ) ) ) ) / ( vf_gridWidth_z * vf_gridWidth_z ) ) )),
            ((1, 0, 0), ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x + ( 0.5 * vf_gridWidth_x ) ), vf_nodePosition_y, vf_nodePosition_z ) ) / ( vf_gridWidth_x * vf_gridWidth_x ) )),
            ((-1, 0, 0), ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x - ( 0.5 * vf_gridWidth_x ) ), vf_nodePosition_y, vf_nodePosition_z ) ) / ( vf_gridWidth_x * vf_gridWidth_x ) )),
            ((0, 1, 0), ( ( -1.0 * getCoefficient ( vf_nodePosition_x, ( vf_nodePosition_y + ( 0.5 * vf_gridWidth_y ) ), vf_nodePosition_z ) ) / ( vf_gridWidth_y * vf_gridWidth_y ) )),
            ((0, -1, 0), ( ( -1.0 * getCoefficient ( vf_nodePosition_x, ( vf_nodePosition_y - ( 0.5 * vf_gridWidth_y ) ), vf_nodePosition_z ) ) / ( vf_gridWidth_y * vf_gridWidth_y ) )),
            ((0, 0, 1), ( ( -1.0 * getCoefficient ( vf_nodePosition_x, vf_nodePosition_y, ( vf_nodePosition_z + ( 0.5 * vf_gridWidth_z ) ) ) ) / ( vf_gridWidth_z * vf_gridWidth_z ) )),
            ((0, 0, -1), ( ( -1.0 * getCoefficient ( vf_nodePosition_x, vf_nodePosition_y, ( vf_nodePosition_z - ( 0.5 * vf_gridWidth_z ) ) ) ) / ( vf_gridWidth_z * vf_gridWidth_z ) ))
        ]
        return constant.Stencil(entries)

    def generate_exa3(self, name):
        return """        
Globals {
    Val kappa : Real = 10.0
}

Function getCoefficient ( xPos : Real, yPos : Real, zPos : Real ) : Real {
    return exp ( ( kappa * ( ( ( xPos - ( xPos ** 2 ) ) * ( yPos - ( yPos ** 2 ) ) ) * ( zPos - ( zPos ** 2 ) ) ) ) )
}

Operator A from Stencil {
    [0, 0, 0] => ( ( ( ( getCoefficient ( ( vf_nodePosition_x@current + ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current, vf_nodePosition_z@current ) + getCoefficient ( ( vf_nodePosition_x@current - ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current, vf_nodePosition_z@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) ) + ( ( getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current + ( 0.5 * vf_gridWidth_y@current ) ), vf_nodePosition_z@current ) + getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current - ( 0.5 * vf_gridWidth_y@current ) ), vf_nodePosition_z@current ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) ) ) + ( ( getCoefficient ( vf_nodePosition_x@current, vf_nodePosition_y@current, ( vf_nodePosition_z@current + ( 0.5 * vf_gridWidth_z@current ) ) ) + getCoefficient ( vf_nodePosition_x@current, vf_nodePosition_y@current, ( vf_nodePosition_z@current - ( 0.5 * vf_gridWidth_z@current ) ) ) ) / ( vf_gridWidth_z@current * vf_gridWidth_z@current ) ) )
    [1, 0, 0] => ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x@current + ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current, vf_nodePosition_z@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) )
    [-1, 0, 0] => ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x@current - ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current, vf_nodePosition_z@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) )
    [0, 1, 0] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current + ( 0.5 * vf_gridWidth_y@current ) ), vf_nodePosition_z@current ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) )
    [0, -1, 0] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current - ( 0.5 * vf_gridWidth_y@current ) ), vf_nodePosition_z@current ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) )
    [0, 0, 1] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, vf_nodePosition_y@current, ( vf_nodePosition_z@current + ( 0.5 * vf_gridWidth_z@current ) ) ) ) / ( vf_gridWidth_z@current * vf_gridWidth_z@current ) )
    [0, 0, -1] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, vf_nodePosition_y@current, ( vf_nodePosition_z@current - ( 0.5 * vf_gridWidth_z@current ) ) ) ) / ( vf_gridWidth_z@current * vf_gridWidth_z@current ) )
}
"""


class MultilinearInterpolationGenerator:

    def __init__(self, coarsening_factor):
        self.coarsening_factor = coarsening_factor

    def generate_stencil(self, grid):
        import lfa_lab as lfa
        from evostencils.model_based_prediction.convergence import lfa_sparse_stencil_to_constant_stencil
        lfa_grid = lfa.Approximation(grid.dimension, grid.spacing)
        lfa_interpolation = lfa.gallery.ml_interpolation_stencil(lfa_grid, lfa_grid.coarse(self.coarsening_factor))
        return lfa_sparse_stencil_to_constant_stencil(lfa_interpolation)

    @staticmethod
    def generate_exa3(name):
        return f"Operator {name} from default prolongation on Node with 'linear'\n"


class FullWeightingRestrictionGenerator:

    def __init__(self, coarsening_factor):
        self.coarsening_factor = coarsening_factor

    def generate_stencil(self, grid):
        import lfa_lab as lfa
        from evostencils.model_based_prediction.convergence import lfa_sparse_stencil_to_constant_stencil
        lfa_grid = lfa.Approximation(grid.dimension, grid.spacing)
        lfa_restriction = lfa.gallery.fw_restriction_stencil(lfa_grid, lfa_grid.coarse(self.coarsening_factor))
        return lfa_sparse_stencil_to_constant_stencil(lfa_restriction)

    @staticmethod
    def generate_exa3(name):
        return f"Operator {name} from default restriction on Node with 'linear'\n"


class IdentityGenerator:
    def __init__(self, dimension):
        self.dimension = dimension

    @staticmethod
    def generate_stencil(grid):
        return constant.get_unit_stencil(grid)

    def generate_exa3(self, name):
        result = f'Operator {name} from Stencil {{\n'
        result += '\t['
        for i in range(self.dimension-1):
            result += '0, '
        result += f'0] => (1.0)\n'
        result += f'}}\n'
        return result


class ZeroGenerator:
    def __init__(self, dimension):
        self.dimension = dimension

    @staticmethod
    def generate_stencil(grid):
        return constant.get_null_stencil(grid)

    def generate_exa3(self, name):
        # TODO implement
        raise NotImplementedError("ZeroGenerator not implemented yet!")
