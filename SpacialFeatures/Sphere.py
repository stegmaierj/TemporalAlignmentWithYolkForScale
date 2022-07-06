import numpy as np

class Sphere:
    """
    A sphere, described via middle point and radius.
    """

    def __init__(
        self,
        middlepoint,
        radius,
        direction,
        pole_1,
        pole_2,
        distance_of_embryo_from_pole,
    ):
        self.middlepoint = middlepoint
        self.radius = radius
        self.direction = direction
        self.pole_1 = pole_1
        self.pole_2 = pole_2 
        self.distance_of_embryo_from_pole = distance_of_embryo_from_pole

    #--------------------------------------------------------------------
    def fromDict(dict):
        return Sphere(np.array(dict['middle_point']), dict['radius'], np.array(dict['direction']), np.array(dict['pole_1']), np.array(dict['pole_2']),dict['distance_of_embryo_from_pole'])
    #--------------------------------------------------------------------

    def coordinateTransform(self, transform):
        raise NotImplementedError()
    #--------------------------------------------------------------------

    def __hash__(self):
        """
        Returns a hash code for the class object
        """
        return hash(
            (
                self.middlepoint,
                self.radius,
                self.direction,
                self.pole_1,
                self.pole_2,
                self.distance_of_embryo_from_pole,
            )
        )
    #--------------------------------------------------------------------

    def __eq__(self, other):
        """
        Checks weather two objects are equal or not
        """
        if not isinstance(other, Sphere):
            return False
        else:
            return (
                self.middlepoint,
                self.radius,
                self.direction,
                self.pole_1,
                self.pole_2,
                self.distance_of_embryo_from_pole,
            ) == (
                other.middlepoint,
                other.radius,
                self.direction,
                self.pole_1,
                self.pole_2,
                self.distance_of_embryo_from_pole,
            )
    #--------------------------------------------------------------------

    def dir_representation(self):
        return {'radius': self.radius,
                    'middle_point': self.middlepoint.tolist(),
                    'pole_1': self.pole_1.tolist(),
                    'pole_2': self.pole_2.tolist(),
                    'direction': self.direction.tolist(),
                    'distance_of_embryo_from_pole': self.distance_of_embryo_from_pole.tolist()}