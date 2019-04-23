
class InitializationInformation:

    @staticmethod
    def get_boundary_as_str():
        return '( 1.0 - exp ( ( ( -1.0 * kappa ) * ( ( vf_boundaryCoord_x@current - ( vf_boundaryCoord_x@current ** 2 ) ) * ( vf_boundaryCoord_y@current - ( vf_boundaryCoord_y@current ** 2 ) ) ) ) ) )'

    @staticmethod
    def get_rhs_as_str():
        return '( ( 2.0 * kappa ) * ( ( vf_nodePosition_x@current - ( vf_nodePosition_x@current ** 2 ) ) + ( vf_nodePosition_y@current - ( vf_nodePosition_y@current ** 2 ) ) ) )'
