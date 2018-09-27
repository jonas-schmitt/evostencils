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
