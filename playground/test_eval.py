from evostencils.expressions.base import *
from evostencils.expressions.multigrid import *

A = Operator('A', (1,1))
f = Grid('f', 1)
u = Grid('u', 1)
I = Identity((1,1))
correct = Correction(I, u, A, f)
str = repr(correct)
print(str)
bar = eval(str)
print(repr(bar))
