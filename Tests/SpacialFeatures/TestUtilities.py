import numpy as np


class TestUtilities:
    @staticmethod
    def create_sphere_test_data(
        percentage_epiboly, radius, middle_point, number_of_points
    ):

        thikness = 30

        #in spherical coordinates:
        radii = np.random.uniform(
            radius-thikness/2, radius+thikness/2, number_of_points)

        #angles:
        phi = np.random.uniform(
            0, 2*np.pi, number_of_points)
        
        rho_max = 1/4*2*np.pi*radius*(radius/(percentage_epiboly*2))
        rho = np.random.uniform(
            0, rho_max, number_of_points)
    
        # random x value
        #upper_bound_x = radius
        #lower_bound_x = upper_bound_x - (radius * 2 * percentage_epiboly)
        x_values = radii*np.cos(phi)*np.sin(rho)
        # random y value
        y_values = radii*np.sin(phi)*np.sin(rho)
        z_values = radii*np.cos(phi)
        pc = {
            "x": x_values + middle_point[0],
            "y": y_values + middle_point[1],
            "z": z_values + middle_point[2],
        }
        return pc

    @staticmethod
    def create_4d_sphere_test_data(
        percentage_epiboly, radius, middle_point, number_of_points
    ):
        pc_first_time_step = TestUtilities.create_sphere_test_data(
        percentage_epiboly, radius, middle_point, number_of_points
        )
        pc_first_time_step['id'] = np.arange(0,pc_first_time_step['x'].size)
        pc_first_time_step['time'] = np.ones(pc_first_time_step['x'].size)
        pc_first_time_step['mother_id'] = np.full(pc_first_time_step['x'].size, -1)

        # next step:

        #rotate points a random angle:

        #random angle x direction:
        x_angle = np.random.uniform(0, 1/4 * np.pi)
        #random angle y direction:
        y_angle = np.random.uniform(0, 1/4 * np.pi)
        #random angle z direction:
        z_angle = np.random.uniform(0, 1/4 * np.pi)

        #rotation matrix:
        R = np.matmul(
            np.matmul(
                [
                    [1.0, 0.0, 0.0],
                    [0, np.cos(x_angle), -np.sin(x_angle)],
                    [0, np.sin(x_angle), np.cos(x_angle)],
                ],
                [
                    [np.cos(y_angle), 0.0, np.sin(y_angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(y_angle), 0.0, np.cos(y_angle)],
                ],
            ),
            [
                [np.cos(z_angle), -np.sin(z_angle), 0],
                [np.sin(z_angle), np.cos(z_angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

        next_points = R.dot(np.array([pc_first_time_step['x'], pc_first_time_step['y'], pc_first_time_step['z']]))
        next_ids = pc_first_time_step['id'] + np.max(pc_first_time_step['id'])
        next_mother_ids = pc_first_time_step['id']
        next_time = pc_first_time_step['time'] + 1

        pc = {}
        pc['id'] = np.concatenate((pc_first_time_step['id'], next_ids))
        pc['mother_id'] = np.concatenate((pc_first_time_step['mother_id'], next_mother_ids))
        pc['time'] = np.concatenate((pc_first_time_step['time'], next_time))
        pc['x'] = np.concatenate((pc_first_time_step['x'], next_points[0,:]))
        pc['y'] = np.concatenate((pc_first_time_step['y'], next_points[1,:]))
        pc['z'] = np.concatenate((pc_first_time_step['z'], next_points[1,:]))
        return pc


