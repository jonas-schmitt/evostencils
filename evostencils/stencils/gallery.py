from evostencils.stencils import constant


def generate_poisson_1d(grid):
    h, = grid.step_size
    entries = [
        ((-1,), -1 / (h * h)),
        ((0,), 2 / (h * h)),
        ((1,), -1 / (h * h)),
    ]
    return constant.Stencil(entries)


def generate_poisson_2d(grid):
    eps = 1.0
    h0, h1 = grid.step_size
    entries = [
        ((0, -1), -1 / (h1 * h1)),
        ((-1, 0), -1 / (h0 * h0) * eps),
        ((0, 0), 2 / (h0 * h0) * eps + 2 / (h1 * h1)),
        ((1, 0), -1 / (h0 * h0) * eps),
        ((0, 1), -1 / (h1 * h1))
    ]
    return constant.Stencil(entries)


def generate_poisson_2d_with_variable_coefficients(grid, get_coefficient: callable, position: tuple):
    assert len(position) == 2, 'Position must be a two dimensional array'
    pos_x = position[0]
    pos_y = position[1]
    width_x, width_y = grid.step_size
    entries = [
        ((0, 0), (((get_coefficient((pos_x + (0.5 * width_x)), pos_y) + get_coefficient((pos_x - (0.5 * width_x)), pos_y))
                   / (width_x * width_x))
                  + ((get_coefficient(pos_x, (pos_y + (0.5 * width_y))) + get_coefficient(pos_x, (pos_y - (0.5 * width_y))))
                     / (width_y * width_y)))),
        ((1, 0), ((-1.0 * get_coefficient((pos_x + (0.5 * width_x)), pos_y))
                  / (width_x * width_x))),
        ((-1, 0), ((-1.0 * get_coefficient((pos_x - (0.5 * width_x)), pos_y))
                   / (width_x * width_x))),
        ((0, 1),  ((-1.0 * get_coefficient(pos_x, (pos_y + (0.5 * width_y))))
                   / (width_y * width_y))),
        ((0, -1), ((-1.0 * get_coefficient(pos_x, (pos_y - (0.5 * width_y))))
                   / (width_y * width_y)))
    ]
    return constant.Stencil(entries)
