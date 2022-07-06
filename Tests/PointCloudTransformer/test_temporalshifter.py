

import numpy as np

from PointCloudTransformer.TemporalShifter import TemporalShifter



def test_that_temporal_shifter_correctly_adjusts_mother_ids():
    x = np.array([1,2,3,4,5,6,7,8,9])
    y = np.array([1,2,3,4,5,6,7,8,9])
    z = np.array([1,2,3,4,5,6,7,8,9])
    id = np.array([1,2,3,4,5,6,7,8,9])
    mother_id = np.array([9,8,9,6,1,5,3,3,-1])
    time = np.array([2,4,2,5,3,4,3,3,1])
    pc = {'x': x, 'y': y, 'z': z, 'mother_id': mother_id, 'id': id, 'time': time }
    shift_map = np.array([[2,1],[4,2],[5,3]])
    expected_time_after_shift = np.array([1,2,1,3,2])
    expected_x_after_shift = np.array([1,2,3,4,6])
    expected_y_after_shift = np.array([1,2,3,4,6])
    expected_z_after_shift = np.array([1,2,3,4,6])
    expected_id_after_shift = np.array([1,2,3,4,6])
    expected_mother_id_after_shift = np.array([-1,3,-1,6,1])
    shifter = TemporalShifter()
    shifted_pc = shifter.shift(pc, shift_map)
    np.testing.assert_almost_equal(expected_time_after_shift, shifted_pc['time'])
    np.testing.assert_almost_equal(expected_x_after_shift, shifted_pc['x'])
    np.testing.assert_almost_equal(expected_y_after_shift, shifted_pc['y'])
    np.testing.assert_almost_equal(expected_z_after_shift, shifted_pc['z'])
    np.testing.assert_almost_equal(expected_id_after_shift, shifted_pc['id'])
    np.testing.assert_almost_equal(expected_mother_id_after_shift, shifted_pc['mother_id'])


def test_that_temporal_shifter_add_new_frames_with_realistic_coordinates_if_necessary():
    x = np.array([1,2,3,4,5,6,7,8,9])
    y = np.array([1,2,3,4,5,6,7,8,9])
    z = np.array([1,2,3,4,5,6,7,8,9])
    id = np.array([1,2,3,4,5,6,7,8,9])
    mother_id = np.array([9,8,9,6,1,5,3,3,-1])
    time = np.array([2,4,2,5,3,4,3,3,1])
    pc = {'x': x, 'y': y, 'z': z, 'mother_id': mother_id, 'id': id, 'time': time }
    shift_map = np.array([[2,1],[3,3]])
    expected_time_after_shift = np.array([1,1,3,3,3,2,2,2])
    expected_x_after_shift = np.array([1,3,5,7,8,3,5,5.5])
    expected_y_after_shift = np.array([1,3,5,7,8,3,5,5.5])
    expected_z_after_shift = np.array([1,3,5,7,8,3,5,5.5])
    expected_id_after_shift = np.array([1,3,5,7,8,10,11,12])
    expected_mother_id_after_shift = np.array([-1,-1,10,11,12,1,3,3])
    shifter = TemporalShifter()
    shifted_pc = shifter.shift(pc, shift_map)
    np.testing.assert_almost_equal(expected_time_after_shift, shifted_pc['time'])
    np.testing.assert_almost_equal(expected_x_after_shift, shifted_pc['x'])
    np.testing.assert_almost_equal(expected_y_after_shift, shifted_pc['y'])
    np.testing.assert_almost_equal(expected_z_after_shift, shifted_pc['z'])
    np.testing.assert_almost_equal(expected_id_after_shift, shifted_pc['id'])
    np.testing.assert_almost_equal(expected_mother_id_after_shift, shifted_pc['mother_id'])


def test_that_temporal_shifter_add_several_new_frames_with_realistic_coordinates_if_necessary():
    x = np.array([1,2,3,4,5,6,7,8,9])
    y = np.array([1,2,3,4,5,6,7,8,9])
    z = np.array([1,2,3,4,5,6,7,8,9])
    id = np.array([1,2,3,4,5,6,7,8,9])
    mother_id = np.array([9,8,9,6,1,5,3,3,-1])
    time = np.array([2,4,2,5,3,4,3,3,1])
    pc = {'x': x, 'y': y, 'z': z, 'mother_id': mother_id, 'id': id, 'time': time }
    shift_map = np.array([[2,1],[3,4]])
    expected_time_after_shift = np.array([1,1,4,4,4,3,3,3,2,2,2])
    expected_x_after_shift = np.array([1,3,5,7,8,1+8/3,3+8/3,3+10/3,1+4/3,3+4/3,3+5/3])
    expected_y_after_shift = np.array([1,3,5,7,8,1+8/3,3+8/3,3+10/3,1+4/3,3+4/3,3+5/3])
    expected_z_after_shift = np.array([1,3,5,7,8,1+8/3,3+8/3,3+10/3,1+4/3,3+4/3,3+5/3])
    expected_id_after_shift = np.array([1,3,5,7,8,10,11,12,13,14,15])
    expected_mother_id_after_shift = np.array([-1,-1,10,11,12,13,14,15,1,3,3])
    shifter = TemporalShifter()
    shifted_pc = shifter.shift(pc, shift_map)
    np.testing.assert_almost_equal(expected_time_after_shift, shifted_pc['time'])
    np.testing.assert_almost_equal(expected_x_after_shift, shifted_pc['x'])
    np.testing.assert_almost_equal(expected_y_after_shift, shifted_pc['y'])
    np.testing.assert_almost_equal(expected_z_after_shift, shifted_pc['z'])
    np.testing.assert_almost_equal(expected_id_after_shift, shifted_pc['id'])
    np.testing.assert_almost_equal(expected_mother_id_after_shift, shifted_pc['mother_id'])