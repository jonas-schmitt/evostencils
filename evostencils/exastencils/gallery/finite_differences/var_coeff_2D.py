def global_declarations():
    return """
Globals {
    Val kappa : Real = 10.0
}
"""


def operator():
    return """
Function getCoefficient ( xPos : Real, yPos : Real ) : Real {
    return exp ( ( kappa * ( ( xPos - ( xPos ** 2 ) ) * ( yPos - ( yPos ** 2 ) ) ) ) )
}

Stencil LaplaceStencil@all{
    [0, 0] => ( ( ( getCoefficient ( ( vf_nodePosition_x@current + ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) + getCoefficient ( ( vf_nodePosition_x@current - ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) ) + ( ( getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current + ( 0.5 * vf_gridWidth_y@current ) ) ) + getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current - ( 0.5 * vf_gridWidth_y@current ) ) ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) ) )
    [1, 0] => ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x@current + ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) )
    [-1, 0] => ( ( -1.0 * getCoefficient ( ( vf_nodePosition_x@current - ( 0.5 * vf_gridWidth_x@current ) ), vf_nodePosition_y@current ) ) / ( vf_gridWidth_x@current * vf_gridWidth_x@current ) )
    [0, 1] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current + ( 0.5 * vf_gridWidth_y@current ) ) ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) )
    [0, -1] => ( ( -1.0 * getCoefficient ( vf_nodePosition_x@current, ( vf_nodePosition_y@current - ( 0.5 * vf_gridWidth_y@current ) ) ) ) / ( vf_gridWidth_y@current * vf_gridWidth_y@current ) )
}
"""

def boundary():
    return '( 1.0 - exp ( ( ( -1.0 * kappa ) * ( ( vf_boundaryCoord_x@current - ( vf_boundaryCoord_x@current ** 2 ) ) * ( vf_boundaryCoord_y@current - ( vf_boundaryCoord_y@current ** 2 ) ) ) ) ) )\n'

def rhs():
    return '( ( 2.0 * kappa ) * ( ( vf_nodePosition_x@current - ( vf_nodePosition_x@current ** 2 ) ) + ( vf_nodePosition_y@current - ( vf_nodePosition_y@current ** 2 ) ) ) )\n'