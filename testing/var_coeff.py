from __future__ import division, print_function

from lfa_lab import *
from lfa_lab.plot import *
import lfa_lab.gallery as gallery
import math
import json
import sys
import copy
import os.path
import time
import multiprocessing
import traceback

import numpy as np
import matplotlib.pyplot as plt

def coeff_fun(conf):
    """Construct the coefficient function.

    This function returns the coefficient function of the PDE.
    """

    # coefficient of the exponential rhs
    k = 10

    if conf.coeff_fun == 'constant':
        def f(x,y,z=0):
            return -1

    elif conf.coeff_fun == 'exponential':

        if conf.dimension == 2:
            def f(x,y):
                if (x < 0 and x > 1) or \
                        (y < 0 and y > 1):
                    raise Exception("Function not defined here")
                return -math.exp(k*(x-x**2)*(y-y**2))

        elif conf.dimension == 3:

            def f(x,y,z):
                if (x < 0 and x > 1) or \
                        (y < 0 and y > 1) \
                        (z < 0 and z > 1):
                    raise Exception("Function not defined here")
                return -math.exp(k*(x-x**2)*(y-y**2)*(z-z**2))

    else:
        f = conf.coeff_fun

    return f


def make_stencil_2d_at(a,pos,grid):
    """

    a: Coefficient function
    x,y: Position
    h: Step size
    """
    assert(isinstance(grid, Grid))
    assert(len(pos) == 2)

    x,y = pos

    # make it periodic
    x = x % 1.0
    y = y % 1.0

    s = SparseStencil()
    h = grid.step_size()

    s.append( ( 0,-1),  a(x,        y-h[1]/2) / h[1]**2)
    s.append( (-1, 0),  a(x-h[0]/2, y       ) / h[0]**2)
    s.append( ( 1, 0),  a(x+h[0]/2, y       ) / h[0]**2)
    s.append( ( 0, 1),  a(x,        y+h[1]/2) / h[1]**2)

    sum_ = 0
    for o, v in s:
        sum_ += v

    s.append( (0,0), -sum_ )

    return s

def make_stencil_3d_at(a, pos, grid):
    assert(isinstance(grid, Grid))
    assert(len(pos) == 3)

    pos = np.array(pos)

    s = SparseStencil(grid)
    h = np.array(grid.step_size())

    x,y,z = pos

    s.append( ( 0, 0,-1),  a(x,        y       ,z-h[2]/2) / h[2]**2)
    s.append( ( 0,-1, 0),  a(x,        y-h[1]/2,z       ) / h[1]**2)
    s.append( (-1, 0, 0),  a(x-h[0]/2, y       ,z       ) / h[0]**2)
    s.append( ( 1, 0, 0),  a(x+h[0]/2, y       ,z       ) / h[0]**2)
    s.append( ( 0, 1, 0),  a(x,        y+h[1]/2,z       ) / h[1]**2)
    s.append( ( 0, 0, 1),  a(x,        y       ,z+h[2]/2) / h[2]**2)

    sum_ = 0
    for o, v in s:
        sum_ += v

    s.append( (0, 0, 0), -sum_)

    return s


def make_stencil_at(a, pos, grid):
    assert(isinstance(grid, Grid))
    dim = len(pos)

    return { 2: make_stencil_2d_at,
             3: make_stencil_3d_at }[dim](a, pos, grid)

def triangle_function(x):
    p = (x % 2.0)

    if p < 1:
        return p
    else:
        return 2 - p

def saw_function(x):
    return x % 1.0

def linear_transform(f, a, b):
    """
    Map the range of (a, b) to (0, 1) call f and do the reverse transform.
    """
    # print(a, b)

    m = 1/(b-a)
    c = -m * a

    def phi(x):
        return m * x + c

    def phi_inv(y):
        return (y-c)/m

    return lambda x: phi_inv(f(phi(x)))

def single_linear_transform(f, a, b):
    """
    Map the range of (a, b) to (0, 1) call f
    """
    # print(a, b)

    m = 1.0/(b-a)
    c = -m * a

    def phi(x):
        return m * x + c

    def new_function(x):
        t = f(phi(x))
        #print(t)
        return t

    return new_function

# Make a function periodic with Period 2*[b-a] by mirroring and repeating the
# interval [a, b]
def make_periodic_function_2d(f, a, b):
    ax, ay = a
    bx, by = b

    gx = linear_transform(triangle_function, ax, bx)
    gy = linear_transform(triangle_function, ay, by)

    return lambda x,y: f(gx(x), gy(y))

def make_periodic_function_3d(f, a, b):
    ax, ay, az = a
    bx, by, bz = b

    gx = linear_transform(triangle_function, ax, bx)
    gy = linear_transform(triangle_function, ay, by)
    gz = linear_transform(triangle_function, az, bz)

    return lambda x,y,z: f(gx(x), gy(y), gz(z))


def smooth_filter(x):
    assert(x >= 0 and x <= 1)
    if 0 <= x and x < 1/4:
        return 0.5 - 0.5 * math.cos(4 * math.pi * x)
    elif 3/4 <= x and x <= 1:
        return 0.5 - 0.5 * math.cos(4 * math.pi * (1 - x))
    else:
        return 1.0

def make_smooth_periodic_2d(f, a, b):
    ax, ay = a
    bx, by = b

    center_x = (ax + bx)/2
    center_y = (ay + by)/2
    c = f(center_x, center_y)

    gx = single_linear_transform(smooth_filter, ax, bx)
    gy = single_linear_transform(smooth_filter, ay, by)

    def filter(x,y):
        return gx(x)*gy(y)

    def filtered_function(x,y):
        return filter(x,y)*f(x,y) + (1.0-filter(x,y))*c

    def scaled_saw(x,y):
        return (linear_transform(saw_function, ax, bx)(x),
                linear_transform(saw_function, ay, by)(y))

    return lambda x,y: filtered_function(*scaled_saw(x,y))

def make_smooth_periodic_3d(f, a, b):
    ax, ay, az = a
    bx, by, bz = b

    center_x = (ax + bx)/2
    center_y = (ay + by)/2
    center_z = (az + bz)/2
    c = f(center_x, center_y, center_z)

    gx = single_linear_transform(smooth_filter, ax, bx)
    gy = single_linear_transform(smooth_filter, ay, by)
    gz = single_linear_transform(smooth_filter, az, bz)

    def filter(x,y,z):
        return gx(x)*gy(y)*gz(z)

    def filtered_function(*args):
        return filter(*args)*f(*args) + (1.0-filter(*args))*c

    def scaled_saw(x,y,z):
        return (linear_transform(saw_function, ax, bx)(x),
                linear_transform(saw_function, ay, by)(y),
                linear_transform(saw_function, az, bz)(z))

    return lambda x,y,z: filtered_function(*scaled_saw(x,y,z))



def make_variable_coeff_op_2d(a, pos, grid, b):
    assert(isinstance(grid, Grid))

    x,y = pos
    h = grid.step_size()

    S = NdArray(shape=((b,b)))
    for i in range(b):
        for j in range(b):
            S[(i,j)] = make_stencil_at(a, (x+i*h[0], y+j*h[1]), grid)

    # print(S)

    return operator.from_periodic_stencil(S, grid)

def make_variable_stencil_3d(a, pos, grid, b):
    assert(isinstance(grid, Grid))

    h = grid.step_size()
    x,y,z = pos
    S = NdArray(shape=(b, b, b))

    for i in range(b):
        for j in range(b):
            for k in range(b):
                S[(i,j,k)] = make_stencil_at(a, (x+i*h[0], y+j*h[1], z+k*h[2]), grid)

    # print(S)

    return PeriodicStencil(S._entries)

def make_variable_coeff_op(a, pos, grid, b):

    if len(h) == 2:
        return make_variable_coeff_op_2d(a, pos, grid, b)
    elif len(h) == 3:
        return make_variable_coeff_op_3d(a, pos, grid, b)


def mkSmoother(conf, L):

    if (conf.smoother == 'Jac'):
        return jacobi(L, conf.smoother_weight)
    elif (conf.smoother == 'RBGS'):
        return rb_jacobi(L, conf.smoother_weight)

def multi_grid_analysis(conf, coeff_fun, pos, grid, level = 1):
    assert(isinstance(grid, Grid))

    coarsening = np.ones(conf.dimension)*2
    coarse_grid = grid.coarse(coarsening)
    h = grid.step_size()
    h2 = coarse_grid.step_size()

    dim = conf.dimension
    b = conf.block_size

    if level == conf.levels:
        return ZeroNode(grid)
    else:
        assert(b%2 == 0)

        if dim == 2:
            coarsening_factor = (2,2)
        elif dim == 3:
            coarsening_factor = (2,2,2)


        #L = mkFoSplitting(make_stencil_at(coeff_fun, pos, h), h)
        L = make_variable_coeff_op(coeff_fun, pos, grid, b)

        I = IdentityNode(grid)
        I_c = IdentityNode(coarse_grid)

        # FoRef S1 = mkFoGsLex(L);
        #S1 = mkFoJac(L, conf.smoother_weight)
        #S1 = mkFoRbJac(L, omega)
        #S1 = S1*S1*S1
        S = mkSmoother(conf, L)
        S1 = S ** conf.no_pre_smoothing
        S2 = S ** conf.no_post_smoothing

        P = gallery.ml_interpolation(grid, coarse_grid)
        R = gallery.fw_restriction(grid, coarse_grid)

        #R = mkFoAdjoint(P)
        # print(st_R)

        # Lc = R * L.full() * P
        #Lc = mkFoStencil(make_stencil_at(coeff_fun, pos, h2), h2)
        Lc = make_variable_coeff_op(coeff_fun, pos, coarse_grid, b//2)


        E_c = multi_grid_analysis(conf, coeff_fun, pos, coarse_grid, level+1)

        CGC = coarse_grid_correction(
                operator = L,
                coarse_operator = Lc,
                interpolation = P,
                restriction = R,
                coarse_error = E_c)
        CGCs = CGC ** conf.no_cgc

        return S2 * CGCs  * S1

def analyze_at(conf, pos):

    result = Result(conf)
    b = conf.block_size
    h = conf.step_size

    grid = Grid(conf.dimension, h)

    # construct periodic function

    try:
        if len(pos) == 2:
            x,y = pos
            coeff = make_periodic_function_2d(
                    coeff_fun(conf),
                    (x-h[0]*b*0, y-h[1]*b*0),
                    (x+h[0]*b*1/2, y+h[1]*b*1/2))
            #coeff = make_smooth_periodic_2d(
            #        coeff_fun(conf),
            #        (x-h[0]*b*1/2, y-h[1]*b*1/2),
            #        (x+h[0]*b*1/2, y+h[1]*b*1/2))

        elif len(pos) == 3:
            x,y,z = pos
            coeff = make_periodic_function_3d(
                    coeff_fun(conf),
                    (x-h[0]*b*0, y-h[1]*b*0, z-h[2]*b*0),
                    (x+h[0]*b*1/2, y+h[1]*b*1/2, z+h[2]*b*1/2))
            #coeff = make_smooth_periodic_3d(
            #        coeff_fun(conf),
            #        (x-h[0]*b*1/2, y-h[1]*b*1/2, z-h[2]*b*1/2),
            #        (x+h[0]*b*1/2, y+h[1]*b*1/2, z+h[2]*b*1/2))

        #print(x,y)

        #s = make_stencil_at(x, y, h)
        #print(s)

        #L = mkFoSplitting(make_stencil_at(a, x, y, h), h)
        L = make_variable_coeff_op(coeff,pos,grid,b)
        #E = mkFoJac(L, 0.7)
        #H = mkHighPassFilter(h, (2,2))

        E = multi_grid_analysis(conf, coeff, pos, grid)

        # residual reduction
        Res = L * E * L.inverse()
        res_sym = Res.symbol()
        #plot_2d(Res)

        smpl = E.symbol()
        result.convergence_rates = res_sym.spectral_radius()
        result.spectral_radius = smpl.spectral_radius()
        result.operator_norm = smpl.spectral_norm()

        #print_report(E)
        #plt.show()
    except Exception as e:
        print('Exception: {}'.format(e))
        print(traceback.format_exc())

        result.convergence_rates = 'NA'
        result.spectral_radius = 'NA'
        result.operator_norm = 'NA'

    print('r(E) = {0}'.format(result.spectral_radius))
    print('||E|| = {}'.format(result.operator_norm))
    print('rates   = {0}'.format(result.convergence_rates))

    return result

def smoothing_factor_at(pos, conf):
    grid = Grid(conf.dimension)

    if conf.dimension == 2:
        coarse_grid = grid.coarse((2,2))
        F = HpFilterNode(grid, coarse_grid)
    else:
        coarse_grid = grid.coarse((2,2,2))
        F = mkHighPassFilter(grid, coarse_grid)

    L_st = make_stencil_at(coeff_fun(conf), pos, grid)

    S = mkSmoother(conf, L_st)

    E = S * F * S

    smpl = E.symbol(32 * np.ones(grid.dimension()))

    return smpl.spectral_radius()

def optimize_omega(conf, pos):
    """Compute the optimal weight."""

    results = []
    for omega in np.linspace(0.3,1.7,100):
        print('Analyzing omega={0:10}'.format(omega), end='\r')
        sys.stdout.flush()

        conf.smoother_weight = omega
        r = smoothing_factor_at(pos, conf)
        #r = spectral_radius(multi_grid_analysis(const_coeff_fun, pos, h, b,
        #    omega))
        results.append((r, omega))
    print()

    opt_omega = min(results)[1]
    print('optimal omega {0}'.format(opt_omega))


class Configuration:

    def __init__(self):
        self.name = 'default'
        self.dimension = 2
        self.smoother = "Jac"
        self.smoother_weight = 0.8
        self.no_pre_smoothing = 2
        self.no_post_smoothing = 2
        self.no_cgc = 1
        self.levels = 3

        self.coeff_fun = 'constant'
        self.step_size = (1/64.0, 1/64.0)

        self.block_size = 4

    def set_from_json(self, data):
        self.name = data[0]
        self.dimension = data[1]['dimensionality']
        self.smoother = data[2]['l3tmp_smoother']

        # remove quotation from smoother name
        self.smoother = self.smoother.replace('"', '')

        weights = \
                {('Jac', 2): 0.79,
                 ('Jac', 3): 0.85,
                 ('RBGS', 2): 1.16,
                 ('RBGS', 3): 1.19 }
        self.smoother_weight = weights[(self.smoother, self.dimension)]

        self.no_pre_smoothing = data[1]['l3tmp_numPre']
        self.no_post_smoothing = data[1]['l3tmp_numPost']
        self.no_cgc = data[1]['l3tmp_numRecCycleCalls']

        if data[2]['l3tmp_genStencilFields'] == 'false':
            self.coeff_fun = 'constant'
        elif data[2]['l3tmp_genStencilFields'] == 'true':
            self.coeff_fun = 'exponential'
        else:
            raise Exception('l3tmp_genStencilFields has an invalid value')


        coarse_N = data[1]['numCellsPerDimCoarsest'] # * 2
        h = 1/(coarse_N+1)
        self.step_size = tuple(h*np.ones(self.dimension))

    def __str__(self):
        return str(self.__dict__)

    @property
    def filename(self):
        return os.path.join('out', self.name + '.json')

class ConfigurationList:

    def __init__(self):
        self.data = ''

    def load(self, filename):
        with open(filename, 'r') as fp:
            self.data = json.load(fp)

    def __getitem__(self, k):
        conf = Configuration()
        conf.set_from_json(self.data[k])
        return conf

    def __len__(self):
        return len(self.data)

class Result:

    def __init__(self, conf):
        self.configuration = conf.__dict__
        self.spectral_radius = 'NA'
        self.operator_norm = 'NA'
        self.convergence_rates = 'NA'

    def store(self, filename):

        with open(filename, 'w') as fp:
            json.dump(self.__dict__, fp, indent=2, sort_keys=True)


def predict_configuration(conf):
    start_time = time.time()
    print('###### Start configuration')
    print(conf)
    sys.stdout.flush()

    if os.path.isfile(conf.filename):
        pass
        print('Result in file "{0}" already exists'.format(conf.filename))
    else:
        h = conf.step_size

        # steepest position
        pos = 0.5 * np.ones(conf.dimension)
        pos[1] = h[1]*conf.block_size

        result = analyze_at(conf, pos)
        result.store(conf.filename)


        end_time = time.time()
        print('Elapsed time: {0}'.format(end_time - start_time))


def predict_configurations(filename):

    configs = ConfigurationList()
    configs.load(filename)

    #pool = multiprocessing.Pool()

    for i in range(len(configs)):
        conf = configs[i]

        # compute in parallel
        #pool.apply_async
        apply(predict_configuration, [conf])


    #pool.close()
    # wait for jobs to succeed
    #pool.join()


if __name__ == '__main__':

    conf = Configuration()
    # optimize_omega(conf, pos)

    h = conf.step_size
    pos = 0.5 * np.ones(conf.dimension)
    pos[1] = h[1]
    analyze_at(conf, pos)

    # Set the configuration parameters to analyze different methods.

    #conf.coeff_fun = 'constant'
    conf.coeff_fun = 'exponential'
    #conf.step_size = (1/16, 1/16)
    #conf.no_pre_smoothing = 3
    #conf.no_post_smoothing = 3

    results = []
    for x in np.linspace(0+h[0], 0.5, 8):
        for y in np.linspace(0+h[1], 0.5, 8):
            results.append(analyze_at(conf, (x, y)))

    max_spectral_radius = max(r.spectral_radius for r in results)
    max_operator_norm = max(r.operator_norm for r in results)

    print('max r(E) = {}'.format(max_spectral_radius))
    print('max ||E|| = {}'.format(max_operator_norm))

