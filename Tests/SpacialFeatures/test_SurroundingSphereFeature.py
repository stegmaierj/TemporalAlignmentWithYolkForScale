from SpacialFeatures.SurroundingSphereFeature import SurroundingSphereFeature
from Tests.SpacialFeatures.TestUtilities import TestUtilities
import numpy as np


def test_that_function_returns_surrounding_sphere():

    pc = TestUtilities.create_sphere_test_data(1, 405, [10, 0, 0], 10000)
    feature = SurroundingSphereFeature(type="4points_calculation")

    sphere = feature.apply(pc)
    assert sphere.radius <= 600 # test just basic functionality, not performance
    assert sphere.radius >= 300
    assert (sphere.middlepoint <= [50, 50, 50]).all()
    assert (sphere.middlepoint >= [-50, -50, -50]).all()


def test_that_function_returns_surrounding_sphere_with_inner_points():

    pc = TestUtilities.create_sphere_test_data(0.3, 405, [0, 0, 0], 10000)
    feature = SurroundingSphereFeature(type="4points_calculation")

    sphere = feature.apply(pc)
    assert sphere.radius <= 600 # test just basic functionality
    assert sphere.radius >= 300
    assert (sphere.middlepoint <= [50, 50, 50]).all()
    assert (sphere.middlepoint >= [-50, -50, -50]).all()


def test_that_poles_are_correcly_found():

    pc = TestUtilities.create_sphere_test_data(0.3, 405, [0, 0, 0], 10000)
    feature = SurroundingSphereFeature(type="optimization")

    sphere = feature.apply(pc)
    assert sphere.pole_1[0] >= -300 # test just basic functionality
    assert sphere.pole_1[0] <= 300
    assert sphere.pole_2[0] >= -300
    assert sphere.pole_2[0] <= 300

    # point 0 0 0 leads to nan errors

def test_that_function_returns_surrounding_sphere_with_optimization_approach():

    pc = TestUtilities.create_sphere_test_data(0.3, 400, [0, 0, 0], 10000)
    feature = SurroundingSphereFeature(type="optimization")
    print('loss for ideal would be {}'.format(np.linalg.norm(400 -np.sqrt((0-pc['x'])**2+(0-pc['y'])**2+(0-pc['z'])**2))))
    sphere = feature.apply(pc)
    assert sphere.radius <= 420
    assert sphere.radius >= 380
    assert (sphere.middlepoint <= [10, 10, 10]).all()
    assert (sphere.middlepoint >= [-10, -10, -10]).all()
