import numpy as np

from SpacialFeatures.SpacialFeatureInterface import SpacialFeatureInterface
from SpacialFeatures.SurroundingSphereFeature import SurroundingSphereFeature

class SizeFeature(SpacialFeatureInterface):
    """
    Feature that describe the size of the point cloud given a sphere with poles in pole direction
    """

    def __init__(self) -> None:
        super().__init__()
        self.type = type

    def apply(self, pc, sphere) -> float:
        direction_norm = sphere.direction / np.linalg.norm(sphere.direction)
        distance_from_pool_measured_in_direction = (direction_norm[0]*(pc['x']- sphere.pole_2[0])+direction_norm[1]*(pc['y']- sphere.pole_2[1])+direction_norm[2]*(pc['z']- sphere.pole_2[2]))/(direction_norm[0]**2+direction_norm[1]**2+direction_norm[2]**2)
        number_to_consider = np.max([(int) (0.02 * pc['x'].size),2])
        max_dictances = np.partition(distance_from_pool_measured_in_direction, -number_to_consider)[-number_to_consider:]
        upper = np.median(max_dictances)
        min_dictances = np.partition(distance_from_pool_measured_in_direction, number_to_consider)[:number_to_consider]
        lower = np.median(min_dictances)
        return upper - lower

    def apply_over_time(self, pointCloud):

        surrSphereFeature = SurroundingSphereFeature(type='optimization')

        time = np.array(pointCloud["time"])
        pc = {
            'x': pointCloud["x"][time==min(time)],
            'y': pointCloud["y"][time==min(time)],
            'z': pointCloud["z"][time==min(time)]
        }
        sphere = surrSphereFeature.apply(pc)

        size = np.full(370,-1, dtype=float)
        number_of_points = np.full(370,-1, dtype=int)
        for i in range(370):
            pc = {
            'x': pointCloud["x"][time==(i+1)],
            'y': pointCloud["y"][time==(i+1)],
            'z': pointCloud["z"][time==(i+1)]
            }
            number_of_points[i] = pc['x'].size
            if number_of_points[i] == 0:
                    size[i] = 0
            else:
                size[i] = self.apply(pc, sphere)
            print('{}/370 done'.format(i))
        return (sphere, size,number_of_points)
