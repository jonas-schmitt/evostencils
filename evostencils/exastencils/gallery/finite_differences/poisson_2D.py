
class Declarations:
    @staticmethod
    def get_globals():
        return ''

    @staticmethod
    def get_operator_stencil():
        return """
    [ 0,  0] =>  2.0 / ( vf_gridWidth_x ** 2 ) + 2.0 / ( vf_gridWidth_y ** 2 )
    [-1,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 1,  0] => -1.0 / ( vf_gridWidth_x ** 2 )
    [ 0, -1] => -1.0 / ( vf_gridWidth_y ** 2 )
    [ 0,  1] => -1.0 / ( vf_gridWidth_y ** 2 )
    """

    @staticmethod
    def get_boundary_as_string():
        return 'cos ( PI * vf_boundaryPos_x ) - sin ( 2.0 * PI * vf_boundaryPos_y )\n'

    @staticmethod
    def get_rhs_as_string():
        return 'PI**2 * cos ( PI * vf_nodePos_x ) - 4.0 * PI**2 * sin ( 2.0 * PI * vf_nodePos_y )\n'
