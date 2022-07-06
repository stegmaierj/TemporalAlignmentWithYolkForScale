from itertools import combinations

import numpy as np
import scipy
import torch
from sklearn.cluster import DBSCAN

from SpacialFeatures.SpacialFeatureInterface import SpacialFeatureInterface
from SpacialFeatures.Sphere import Sphere


class SurroundingSphereFeature(SpacialFeatureInterface):
    """
    Feature that describe a sphere that surrounds most of the point cloud and is as small as possible.
    """

    def __init__(self, type) -> None:
        """possible types:
        '4points_calculation': use several combinations of 4 outer points to calculate sphere
        'optimization': use a binary tree to find ideal radius, and in each step find ideal middle point with Adam Optimizer
        """
        super().__init__()
        self.type = type

#-------------------------------------------------------------------------------


    def apply(self, pointCloud) -> Sphere:

        x = pointCloud["x"].astype(np.float)
        y = pointCloud["y"].astype(np.float)
        z = pointCloud["z"].astype(np.float)
        m = np.array([0,0,0])

        db = DBSCAN(eps=60, min_samples=3).fit(np.array([x, y, z]).transpose())

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        is_no_outlier = labels != -1

        with open('exp2_log.txt', 'a') as log_file:
            log_file.write('number of outliers {}, number of not outliers: {} \n'.format(np.sum(labels==-1), np.sum(labels!=-1)))
        # only take "border" of cluster


        if self.type == "4points_calculation":

            # only take "border" of cluster

            x = x[is_no_outlier]
            y = y[is_no_outlier]
            z = z[is_no_outlier]

            # only take points that are distant to beginning middle point

            for n in range(3):
                mask_up_rigth_back = (x > m[0]) & (y > m[1]) & (z < m[2])
                mask_up_right_front = (x > m[0]) & (y > m[1]) & (z > m[2])
                mask_up_left_back = (x < m[0]) & (y > m[1]) & (z < m[2])
                mask_up_left_front = (x < m[0]) & (y > m[1]) & (z > m[2])

                mask_down_rigth_back = (x > m[0]) & (y < m[1]) & (z < m[2])
                mask_down_right_front = (x > m[0]) & (y < m[1]) & (z > m[2])
                mask_down_left_back = (x < m[0]) & (y < [1]) & (z < m[2])
                mask_down_left_front = (x < m[0]) & (y < m[1]) & (z > m[2])

                area_masks = [
                    mask_up_rigth_back,
                    mask_up_right_front,
                    mask_up_left_back,
                    mask_up_left_front,
                    mask_down_left_back,
                    mask_down_rigth_back,
                    mask_down_left_front,
                    mask_down_right_front,
                ]

                distances = np.sqrt((x - m[0]) ** 2 + (y - m[1]) ** 2 + (z - m[2]) ** 2)

                tuple = np.array([], dtype=int)
                enumeration = np.arange(x.size)
                for mask in area_masks:
                    if np.sum(mask) > 2:
                        tuple = np.append(
                            tuple,
                            enumeration[mask][
                                np.argpartition(distances[mask], -3)[-3:]
                            ],
                        )
                # calculate surrounding sphere with 4 points:

                combination_number = (int)(scipy.special.binom(tuple.size, 4))
                m_sum = np.full([combination_number, 3], -1, dtype=float)
                r_sum = np.full(combination_number, -1, dtype=float)
                loss = np.full(combination_number, -1, dtype=float)
                for (i, tu) in enumerate(combinations(tuple, 4)):
                    tu = np.array(tu)
                    x_values = x[tu]
                    y_values = y[tu]
                    z_values = z[tu]
                    t = -(x_values**2 + y_values**2 + z_values**2)
                    T = np.linalg.det(
                        np.array([x_values, y_values, z_values, np.array([1, 1, 1, 1])])
                    )
                    D = (
                        np.linalg.det(
                            np.array([t, y_values, z_values, np.array([1, 1, 1, 1])])
                        )
                        / T
                    )
                    E = (
                        np.linalg.det(
                            np.array([x_values, t, z_values, np.array([1, 1, 1, 1])])
                        )
                        / T
                    )
                    F = (
                        np.linalg.det(
                            np.array([x_values, y_values, t, np.array([1, 1, 1, 1])])
                        )
                        / T
                    )
                    G = np.linalg.det(np.array([x_values, y_values, z_values, t])) / T
                    m = np.array([-D / 2.0, -E / 2.0, -F / 2.0], dtype=float)
                    m_sum[i] = m
                    radius = 1 / 2 * np.sqrt(D**2 + E**2 + F**2 - 4 * G)
                    loss[i] = np.mean(
                        np.abs(
                            radius
                            - np.sqrt(
                                (m[0] - x[tuple]) ** 2
                                + (m[1] - y[tuple]) ** 2
                                + (m[2] - z[tuple]) ** 2
                            )
                        )
                    )
                    r_sum[i] = radius

                m = m_sum[np.argmin(loss)]
                radius = r_sum[np.argmin(loss)]

        if self.type == "use_tracking_direction":
            raise NotImplementedError()

        if self.type == "optimization":
            x = x[is_no_outlier]
            y = y[is_no_outlier]
            z = z[is_no_outlier]
            with open('exp2_log.txt', 'a') as log_file:
                log_file.write('size x after outlier remoavle: {} \n'.format(x.shape))
            m = np.array([np.mean(x), np.mean(y), np.mean(z)])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            #--------------------------------------------------------------
            # optimize (r, x_m, y_m, z_m), start:
            def optimize_m_with_fixed_r(optimize_param, learnrate,x,y,z,m,radius):
                '''
                Uses Adam optimizer to find ideal middle point with fixed radius
                '''
                x = torch.from_numpy(x).float().to(device)
                y = torch.from_numpy(y).float().to(device)
                z = torch.from_numpy(z).float().to(device)
                m = torch.from_numpy(m).float().to(device)
                m.requires_grad_()
                radius = torch.tensor(radius).to(device)
                radius.requires_grad_()
                if optimize_param == "m":
                    optimize_value = [m]
                if optimize_param == "r":
                    optimize_value = [radius]
                if optimize_param == "m_and_r":
                    optimize_value = [m,radius]
                optimi_data = torch.optim.Adam(optimize_value, lr=learnrate)
                # minimize:
                loss_old = 0
                for i in range(1000000):
                    optimi_data.zero_grad()
                    #use MSE as loss:
                    loss = torch.dist(radius,torch.sqrt((m[0]-x)**2+(m[1]-y)**2+(m[2]-z)**2))
                    if (np.abs(loss.item()-loss_old)< 0.1):
                        break
                    loss_old = loss.item()
                    loss.backward()
                    optimi_data.step()
                m = m.detach().cpu().numpy()
                radius = radius.detach().cpu().item()
                x = x.cpu().numpy()
                y = y.cpu().numpy()
                z = z.cpu().numpy()
                return (m, radius, loss.item())
             #--------------------------------------------------------------

            def optimize(radii, loss_curr, opt_m, opt_radius):
                ''' Optimize m for several fixed radii and return solution with minimal loss
                '''
                for radius_ind in radii:
                    m_after_opt, radius_after_opt, loss = optimize_m_with_fixed_r("m", 0.01,x,y,z,m,radius_ind)
                    print('r: {}, m: {}, loss: {}'.format(m_after_opt, radius_after_opt, loss))
                    if loss < loss_curr:
                        opt_radius = radius_after_opt
                        loss_curr = loss
                        opt_m = m_after_opt
                        print(opt_m, opt_radius, loss_curr)
                return opt_m, opt_radius, loss_curr
            #--------------------------------------------------------------

            def radii_calc(radius, stepsize):
                ''' Use binary search to find optimal radius
                '''
                opt_m, opt_radius, loss_curr = optimize_m_with_fixed_r("m", 0.01,x,y,z,m,radius)
                print('r: {}, m: {}, loss: {}'.format(opt_m,radius, loss_curr))

                while stepsize>1:
                    opt_m, opt_radius, loss_curr = optimize([opt_radius-stepsize, opt_radius+stepsize], loss_curr,opt_m, opt_radius)
                    stepsize = int(stepsize / 2)
                return opt_m, opt_radius, loss_curr
            #--------------------------------------------------------------

            m, radius, loss = radii_calc(400., 100.)

        centre_of_mass = np.array([x, y, z]).mean(axis=1)
        direction = m - centre_of_mass

        t_1 = np.sqrt(
            radius**2 / (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        )
        t_2 = -np.sqrt(
            radius**2 / (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        )
        pole_1 = m + direction * t_1
        pole_2 = m + direction * t_2
        # distance to pool_1 of each point:
        # distance_to_pole_parallel:
        direction_norm = direction / np.linalg.norm(direction)
        distance_from_pool_measured_in_direction = (
            direction_norm[0] * (x - pole_2[0])
            + direction_norm[1] * (y - pole_2[1])
            + direction_norm[2] * (z - pole_2[2])
        ) / (direction_norm[0] ** 2 + direction_norm[1] ** 2 + direction_norm[2] ** 2)
        min_dictances = np.min(distance_from_pool_measured_in_direction)

        return Sphere(np.array(m), radius, direction, pole_1, pole_2, min_dictances)
