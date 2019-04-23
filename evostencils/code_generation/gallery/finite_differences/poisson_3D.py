
class InitializationInformation:

    @staticmethod
    def get_boundary_as_str():
        return 'vf_boundaryCoord_x * vf_boundaryCoord_x - 0.5 * vf_boundaryCoord_y * vf_boundaryCoord_y - 0.5 * vf_boundaryCoord_z * vf_boundaryCoord_z'

    @staticmethod
    def get_rhs_as_str():
        return '0'
