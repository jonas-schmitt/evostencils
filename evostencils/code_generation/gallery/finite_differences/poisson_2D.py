
class InitializationInformation:

    @staticmethod
    def get_boundary_as_str():
        return 'cos ( PI * vf_boundaryPos_x ) - sin ( 2.0 * PI * vf_boundaryPos_y )'

    @staticmethod
    def get_rhs_as_str():
        return 'PI**2 * cos ( PI * vf_nodePos_x ) - 4.0 * PI**2 * sin ( 2.0 * PI * vf_nodePos_y )'
