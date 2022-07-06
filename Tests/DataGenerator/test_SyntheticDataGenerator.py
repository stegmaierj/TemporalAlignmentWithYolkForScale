from DataGenerator.SyntheticDataGenerator import SyntheticDataGenerator
import numpy as np

#-------------------------------------------------------------------------------


def getRandomNumberGeneratorMock(returnValues):
    class randomGeneratorMock():
        def __init__(this):
            this.counter = 0

        def normal(this, loc=0, scale=0, size=(1,)):
            old_counter = this.counter
            this.counter += size[0]
            if size[0] == 1:
                return returnValues[old_counter]
            return np.array(returnValues[old_counter: this.counter ])

        def binomial(this, n=1, p=0.5, size=1):
            old_counter = this.counter
            this.counter += size
            if size == 1:
                return returnValues[old_counter]
            return np.array(returnValues[old_counter: this.counter ])
    return randomGeneratorMock()

#-------------------------------------------------------------------------------


def test_that_synthetic_data_generator_makes_random_scale():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([1.5,2,3,4]), 'y': np.array([3,4,5,6.5]), 'z': np.array([0.3,0,0,0])}

    scaled_pc, scaling_factor = syntheticDataGenerator.scaleRandomly(getRandomNumberGeneratorMock([2.]),pc,1,0.2)
    np.testing.assert_almost_equal(scaled_pc['x'], [3,4,6,8])
    np.testing.assert_almost_equal(scaled_pc['y'], [6,8,10,13])
    np.testing.assert_almost_equal(scaled_pc['z'], [0.6,0,0,0])
    assert scaling_factor == 2.0
    # not in place:
    np.testing.assert_almost_equal(pc['x'], [1.5,2,3,4])
    np.testing.assert_almost_equal(pc['y'], [3,4,5,6.5])
    np.testing.assert_almost_equal(pc['z'], [0.3,0,0,0])

#-------------------------------------------------------------------------------


def test_that_synthetic_data_generator_makes_random_scale_in_place_if_configured():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([1.5,2,3,4]), 'y': np.array([3,4,5,6.5]), 'z': np.array([0.3,0,0,0])}

    scaled_pc, scaling_factor = syntheticDataGenerator.scaleRandomly(getRandomNumberGeneratorMock([2.]),pc,1,0.2, True)
    np.testing.assert_almost_equal(scaled_pc['x'], [3,4,6,8])
    np.testing.assert_almost_equal(scaled_pc['y'], [6,8,10,13])
    np.testing.assert_almost_equal(scaled_pc['z'], [0.6,0,0,0])
    assert scaling_factor == 2.0
    #in place:
    np.testing.assert_almost_equal(pc['x'], [3,4,6,8])
    np.testing.assert_almost_equal(pc['y'], [6,8,10,13])
    np.testing.assert_almost_equal(pc['z'], [0.6,0,0,0])
#-------------------------------------------------------------------------------


def test_that_synthetic_data_generator_makes_random_translate():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([1.5,2,3,4]), 'y': np.array([3,4,5,6.5]), 'z': np.array([0.3,0,0,0])}
    shifted_pc, shift_values = syntheticDataGenerator.shiftRandomly(getRandomNumberGeneratorMock([0,1.1,-90]), pc,0,40)
    np.testing.assert_almost_equal(shifted_pc['x'], [1.5,2,3,4])
    np.testing.assert_almost_equal(shifted_pc['y'], [4.1,5.1,6.1,7.6])
    np.testing.assert_almost_equal(shifted_pc['z'], [-89.7,-90,-90,-90])
    np.testing.assert_almost_equal(shift_values,[0,1.1,-90] )

    #not in place:
    np.testing.assert_almost_equal(pc['x'], [1.5,2,3,4])
    np.testing.assert_almost_equal(pc['y'], [3,4,5,6.5])
    np.testing.assert_almost_equal(pc['z'], [0.3,0,0,0])
#-------------------------------------------------------------------------------

def test_that_synthetic_data_generator_makes_random_translate_inplace_if_configured():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([1.5,2,3,4]), 'y': np.array([3,4,5,6.5]), 'z': np.array([0.3,0,0,0])}
    shifted_pc, shift_values = syntheticDataGenerator.shiftRandomly(getRandomNumberGeneratorMock([0,1.1,-90]), pc,0,40, True)
    np.testing.assert_almost_equal(shifted_pc['x'], [1.5,2,3,4])
    np.testing.assert_almost_equal(shifted_pc['y'], [4.1,5.1,6.1,7.6])
    np.testing.assert_almost_equal(shifted_pc['z'], [-89.7,-90,-90,-90])
    np.testing.assert_almost_equal(shift_values,[0,1.1,-90] )

    #in place:
    np.testing.assert_almost_equal(pc['x'], [1.5,2,3,4])
    np.testing.assert_almost_equal(pc['y'], [4.1,5.1,6.1,7.6])
    np.testing.assert_almost_equal(pc['z'], [-89.7,-90,-90,-90])

#-------------------------------------------------------------------------------


def test_that_synthetic_data_generator_makes_random_rotation_on_y_axes():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([1]), 'y': np.array([0]), 'z': np.array([0])}
    shifted_pc, shift_values = syntheticDataGenerator.rotateRandomly(getRandomNumberGeneratorMock([0, 1./2*np.pi,0]), pc,0,40)
    np.testing.assert_almost_equal(shifted_pc['x'], [0])
    np.testing.assert_almost_equal(shifted_pc['y'], [0])
    np.testing.assert_almost_equal(shifted_pc['z'], [-1])
    np.testing.assert_almost_equal(shift_values,[0, 1./2*np.pi,0] )

    #not in place:
    np.testing.assert_almost_equal(pc['x'], [1])
    np.testing.assert_almost_equal(pc['y'], [0])
    np.testing.assert_almost_equal(pc['z'], [0])

    #-------------------------------------------------------------------------------


def test_that_synthetic_data_generator_makes_random_rotation_on_x_axes():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([0]), 'y': np.array([0]), 'z': np.array([1])}
    shifted_pc, shift_values = syntheticDataGenerator.rotateRandomly(getRandomNumberGeneratorMock([1./2*np.pi,0,0]), pc,0,40)
    np.testing.assert_almost_equal(shifted_pc['x'], [0])
    np.testing.assert_almost_equal(shifted_pc['y'], [-1])
    np.testing.assert_almost_equal(shifted_pc['z'], [0])
    np.testing.assert_almost_equal(shift_values,[1./2*np.pi,0,0] )

    #not in place:
    np.testing.assert_almost_equal(pc['x'], [0])
    np.testing.assert_almost_equal(pc['y'], [0])
    np.testing.assert_almost_equal(pc['z'], [1])
#-------------------------------------------------------------------------------


def test_that_it_createSyntheticSampleWithTranslateAndScale_with_use_of_middle_point():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([1.5,2,3,4]), 'y': np.array([3,4,5,6.5]), 'z': np.array([0.3,0,0,0])}
    middle_point = np.array([1,2,3])
    random_generator = getRandomNumberGeneratorMock([2.,0,1.1,-90])
    shifted_pc, scale_factor, shift_values = syntheticDataGenerator.createSyntheticSampleWithTranslateAndScale(random_generator, pc, middle_point, [1,0.1], [0,30])
    assert scale_factor == 2.0
    np.testing.assert_almost_equal(shift_values,[0,1.1,-90] )
    # minus middlepoint: x: [1.5,2,3,4] - 1 = [0.5,1,2,3]
    # minus middlepoint: y: [3,4,5,6.5] - 2 = [1,2,3,4.5]
    # minus middlepoint: z: [0.3,0,0,0] - 3 = [-2.7,-3,-3,-3]
    # scale with factor 2: x : [1,2,4,6], y: [2,4,6,9], z: [-5.4,-6,-6,-6]
    #translate with [0,1.1,-90]: x: [1,2,4,6], y: [3.1,5.1,7.1,10.1], z: [-95.4,-96,-96,-96]

    np.testing.assert_almost_equal(shifted_pc['x'], [1,2,4,6])
    np.testing.assert_almost_equal(shifted_pc['y'], [3.1,5.1,7.1,10.1])
    np.testing.assert_almost_equal(shifted_pc['z'], [-95.4,-96,-96,-96])

    # not in place:
    np.testing.assert_almost_equal(pc['x'], [1.5,2,3,4])
    np.testing.assert_almost_equal(pc['y'], [3,4,5,6.5])
    np.testing.assert_almost_equal(pc['z'], [0.3,0,0,0])

#-------------------------------------------------------------------------------

def test_that_sythetic_data_generator_adds_noise():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([0.5,1,2,3]), 'y': np.array([1,2,3,4.5]), 'z': np.array([-2.7,-3,-3,-3])}
    random_generator = getRandomNumberGeneratorMock([1,2,3,4,5,6,7,8,-9,-10,11,12])
    shifted_pc = syntheticDataGenerator.scatterPoints(random_generator, pc, 5, False)
    np.testing.assert_almost_equal(shifted_pc['x'], [1.5,3,5,7])
    np.testing.assert_almost_equal(shifted_pc['y'], [6,8,10,12.5])
    np.testing.assert_almost_equal(shifted_pc['z'], [-11.7,-13,8,9])

    # not in place:
    np.testing.assert_almost_equal(pc['x'], [0.5,1,2,3])
    np.testing.assert_almost_equal(pc['y'], [1,2,3,4.5])
    np.testing.assert_almost_equal(pc['z'], [-2.7,-3,-3,-3])

    #-------------------------------------------------------------------------------

def test_that_sythetic_data_generator_adds_noise_in_place():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([0.5,1,2,3]), 'y': np.array([1,2,3,4.5]), 'z': np.array([-2.7,-3,-3,-3])}
    random_generator = getRandomNumberGeneratorMock([1,2,3,4,5,6,7,8,-9,-10,11,12])
    shifted_pc = syntheticDataGenerator.scatterPoints(random_generator, pc, 5, True)
    np.testing.assert_almost_equal(shifted_pc['x'], [1.5,3,5,7])
    np.testing.assert_almost_equal(shifted_pc['y'], [6,8,10,12.5])
    np.testing.assert_almost_equal(shifted_pc['z'], [-11.7,-13,8,9])

    # in place:
    np.testing.assert_almost_equal(pc['x'], [1.5,3,5,7])
    np.testing.assert_almost_equal(pc['y'], [6,8,10,12.5])
    np.testing.assert_almost_equal(pc['z'], [-11.7,-13,8,9])

    #-------------------------------------------------------------------------------


def test_that_sythetic_data_generator_deletes_points():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([0.5,1,2,3]), 'y': np.array([1,2,3,4.5]), 'z': np.array([-2.7,-3,-3,-3]), 'id': np.array([1,2,3,4])}
    random_generator = getRandomNumberGeneratorMock([1, 0, 0, 1])
    shifted_pc = syntheticDataGenerator.deletePoints(random_generator, pc, 0.2, False)
    np.testing.assert_almost_equal(shifted_pc['x'], [1,2])
    np.testing.assert_almost_equal(shifted_pc['y'], [2,3])
    np.testing.assert_almost_equal(shifted_pc['z'], [-3,-3])
    np.testing.assert_almost_equal(shifted_pc['id'], [2,3])

    # not in place:
    np.testing.assert_almost_equal(pc['x'], [0.5,1,2,3])
    np.testing.assert_almost_equal(pc['y'], [1,2,3,4.5])
    np.testing.assert_almost_equal(pc['z'], [-2.7,-3,-3,-3])

    #-------------------------------------------------------------------------------


def test_that_sythetic_data_generator_deletes_points_in_place():
    syntheticDataGenerator = SyntheticDataGenerator()
    pc = {'x': np.array([0.5,1,2,3]), 'y': np.array([1,2,3,4.5]), 'z': np.array([-2.7,-3,-3,-3]), 'id': np.array([1,2,3,4])}
    random_generator = getRandomNumberGeneratorMock([1, 0, 0, 1])
    shifted_pc = syntheticDataGenerator.deletePoints(random_generator, pc, 0.2, True)
    np.testing.assert_almost_equal(shifted_pc['x'], [1,2])
    np.testing.assert_almost_equal(shifted_pc['y'], [2,3])
    np.testing.assert_almost_equal(shifted_pc['z'], [-3,-3])
    np.testing.assert_almost_equal(shifted_pc['id'], [2,3])

    # in place:
    np.testing.assert_almost_equal(pc['x'], [1,2])
    np.testing.assert_almost_equal(pc['y'], [2,3])
    np.testing.assert_almost_equal(pc['z'], [-3,-3])
    np.testing.assert_almost_equal(pc['id'], [2,3])