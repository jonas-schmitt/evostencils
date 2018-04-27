import sympy as sp
import numpy as np
from pystencils.finitedifferences import *
from pystencils import Field
from pystencils.display_utils import to_dot
from pystencils.cpu import create_kernel, make_python_function
from lbmpy.session import *
sp.init_printing()
size = (100,100)
dx = 1
dim = len(size)

# create arrays
uArrays = [np.zeros(size), np.zeros(size)]
uFields = [ Field.create_from_numpy_array("u^%s" % (name,),arr)
            for name, arr in zip(["current", "next"],uArrays)]
pde = diffusion(uFields[0].center, 1)
#pde
#[x.field for x in pde.atoms() if isinstance(x, Field.Access)]
#toDot(pde)
#pde = diffusion(u.center, 1)
disc = Discretization2ndOrder()
discretization = disc(pde).expand()
discretization = discretization.subs(sp.Symbol("dx"), dx)
update = sp.solve(discretization, uFields[0].center)
u_next_C = uFields[1].center
ast = create_kernel([sp.Eq(u_next_C, update[0])])
kernel = ast.compile()
show_code(ast)
X,Y = np.meshgrid( np.linspace(0, 1, size[1]), np.linspace(0,1, size[0]))
#Z = np.sin(2*X*np.pi) * np.sin(2*Y*np.pi)
#Z = np.random.rand(size[0], size[1]) * 2 - 1
Z = np.ones(size)
np.copyto(uArrays[0], Z)


def boundary_handling(u):
    # No concentration at the upper, lower wall and the left inflow border
    u[:, 0] = 0
    u[0, :] = 0
    u[:, -1] = 0
    u[-1, :] = 0

def run_jacobi(timesteps):
    for t in range(timesteps):
        boundary_handling(uArrays[0])
        kernel(u_current=uArrays[0], u_next=uArrays[1])
        uArrays[0] = uArrays[1]
    return X, Y, uArrays[1]


def run_gauss_seidel(timesteps):
    for t in range(timesteps):
        boundary_handling(uArrays[0])
        kernel(u_current=uArrays[0], u_next=uArrays[0])
    return X, Y, uArrays[0]

from pystencils.jupytersetup import make_surface_plot_animation, display_in_extra_window
ani = make_surface_plot_animation(run_jacobi, frames=1000)
display_in_extra_window(ani)
