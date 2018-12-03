
def global_declarations():
    return ""


def operator():
    return """
Operator Laplace from Stencil {
    [ 0,  0] =>  2.0 / ( vf_gridWidth_x ** 2 ) + 2.0 / ( vf_gridWidth_y ** 2 )
    [-1,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 1,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 0, -1] => -1.0 / ( vf_gridWidth_y ** 2 )
    [ 0,  1] => -1.0 / ( vf_gridWidth_y ** 2 )
}
"""


def boundary():
    return 'cos ( PI * vf_boundaryPos_x ) - sin ( 2.0 * PI * vf_boundaryPos_y )\n'


def rhs():
    return 'PI**2 * cos ( PI * vf_nodePos_x ) - 4.0 * PI**2 * sin ( 2.0 * PI * vf_nodePos_y )\n'
