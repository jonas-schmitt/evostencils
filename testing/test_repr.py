from evostencils.expressions.base import *
from evostencils.expressions.multigrid import *
from evostencils.expressions import transformations

A = Operator('A', (1,1))
f = Grid('f', (1,))
u = Grid('u', (1,))
I = Identity((1, 1))
correct = Correction(I, u, A, f)
str = repr(correct)
print(str)
bar = eval(str)
print(repr(bar))
I = Identity((1,1))
Z = Zero((1,1))
tmp = Multiplication(I, Subtraction(Multiplication(I, Z), Z))
#tmp = transformations.remove_identity_operations(tmp)
tmp = transformations.propagate_zero(tmp)
#tmp = transformations.remove_identity_operations(tmp)
print(tmp)
