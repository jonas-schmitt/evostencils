import pygraphviz as pgv

G = pgv.AGraph()
table_obj_nodeid = dict()


# function to convert to superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


# function to convert to subscript
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


def getoperatorsymbol(name: str):
    if name == "Multiplication":
        return "*"
    elif name == "Addition":
        return "+"
    elif name == "Subtraction":
        return "-"


def getapproxsymbol(approximation):
    lvl = str(approximation.grid[0].level)
    itr = 0
    while type(approximation).__name__ != "ZeroApproximation":
        approximation = approximation.approximation
        itr += 1

    itr = f"({itr})"
    return f"{approximation.name}{get_sub(lvl)}{get_super(itr)}"


def create_graph(obj, prev_id=0, next_id=1):
    objtype = type(obj).__name__
    node_id = table_obj_nodeid.get(obj)
    if objtype == "Cycle":
        if node_id is None:
            approximation_id = next_id
            next_id = next_id + 1
            operator_id = next_id
            G.add_node(operator_id, label="+")
            G.add_node(approximation_id, label=getapproxsymbol(obj.approximation))
            if prev_id is not 0:
                G.add_edge(prev_id, operator_id)
            G.add_edge(operator_id, approximation_id)
            next_id = create_graph(obj.correction, operator_id, next_id + 1)
            table_obj_nodeid[obj] = operator_id
        else:
            G.add_edge(prev_id, node_id)
    elif "Approximation" in objtype or "Diagonal" in objtype or "CoarseGridSolver" in objtype or "RightHandSide" in objtype \
            or "Inverse" == objtype or "Operator" == objtype or "Restriction" == objtype or "Prolongation" == objtype:
        operator_id = next_id
        if "Approximation" in objtype:
            obj_label = getapproxsymbol(obj)
        elif "RightHandSide" in objtype or "Operator" == objtype:
            obj_label = f"{str(obj)}{get_sub(str(obj.grid[0].level))}"
        else:
            obj_label = str(obj)
        G.add_node(operator_id, label=obj_label)
        G.add_edge(prev_id, operator_id)
    elif objtype == "Addition" or objtype == "Multiplication" or objtype == "Subtraction":
        if node_id is None:
            opsymbol = getoperatorsymbol(objtype)
            operator_id = next_id
            G.add_node(operator_id, label=opsymbol)
            G.add_edge(prev_id, operator_id)
            next_id = create_graph(obj.operand1, operator_id, next_id + 1)
            next_id = create_graph(obj.operand2, operator_id, next_id + 1)
            table_obj_nodeid[obj] = operator_id
        else:
            G.add_edge(prev_id, node_id)

    elif objtype == "Residual":
        operator_id_sub = next_id
        next_id += 1
        operator_id_mul = next_id
        next_id += 1
        operator_id_rhs = next_id
        G.add_node(operator_id_sub, label="-")
        G.add_node(operator_id_mul, label="*")
        G.add_edge(prev_id, operator_id_sub)
        next_id = create_graph(obj.rhs, operator_id_sub, operator_id_rhs)
        G.add_edge(operator_id_sub, operator_id_mul)
        next_id = create_graph(obj.operator, operator_id_mul, next_id + 1)
        next_id = create_graph(obj.approximation, operator_id_mul, next_id + 1)

    return next_id


def save_graph():
    G.draw("graph.png", prog="dot")
