import pygraphviz as pgv
from evostencils.expressions.base import Cycle, Multiplication, Addition, Subtraction, Residual

G = pgv.AGraph()


def getoperatorsymbol(name: str):
    if name == "Multiplication":
        return "*"
    elif name == "Addition":
        return "+"
    elif name == "Subtraction":
        return "-"


def viz(node_id, obj, ctr=0):
    objtype = type(obj).__name__
    if objtype == "Cycle":
        ctr = ctr + 1
        approximation_id = ctr
        ctr = ctr + 1
        operator_id = ctr
        G.add_node(operator_id, label="+")
        G.add_node(approximation_id, label="u")
        if node_id is not 0:
            G.add_edge(node_id, operator_id)
        G.add_edge(operator_id, approximation_id)
        ctr = viz(operator_id, obj.correction, ctr)
    elif "Approximation" in objtype or "Diagonal" in objtype or "CoarseGridSolver" in objtype \
            or "Inverse" == objtype or "Operator" == objtype or "Restriction" == objtype or "Prolongation" == objtype:
        ctr = ctr + 1
        operator_id = ctr
        G.add_node(operator_id, label=str(obj))
        G.add_edge(node_id, operator_id)
    elif objtype == "Addition" or objtype == "Multiplication" or objtype == "Subtraction":
        opsymbol = getoperatorsymbol(objtype)
        ctr = ctr + 1
        operator_id = ctr
        G.add_node(operator_id, label=opsymbol)
        G.add_edge(node_id, operator_id)
        ctr = viz(operator_id, obj.operand1, ctr)
        ctr = viz(operator_id, obj.operand2, ctr)
    elif objtype == "Residual":
        ctr = ctr + 1
        operator_id_sub = ctr
        ctr = ctr + 1
        operator_id_rhs = ctr
        ctr = ctr + 1
        operator_id_mul = ctr
        G.add_node(operator_id_sub, label="-")
        G.add_node(operator_id_rhs, label="f")
        G.add_node(operator_id_mul, label="*")
        G.add_edge(node_id, operator_id_sub)
        G.add_edge(operator_id_sub, operator_id_rhs)
        G.add_edge(operator_id_sub, operator_id_mul)
        ctr = viz(operator_id_mul, obj.operator, ctr)
        ctr = viz(operator_id_mul, obj.approximation, ctr)

    return ctr


def save():
    G.draw("graph.png", prog="dot")
