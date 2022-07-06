import numpy as np 

from Alignment.SimpleTimeAlign import SimpleTimeAlign

def test_that_simple_align_aligns_with_1d_data_in_simple_case():
    simpleAlign = SimpleTimeAlign()
    unaligned_features_1 = np.array([1,2,3,4,5.1,6,7,8,9,10,11,12,13,14,15,15,15,15,15,16,17,18,18]) 
    unaligned_features_2 = np.array([4.1,5,6,6,7,8,10,11,12,13,14,14,14,14,15,16,17,18,19,20,21,22]) 
    expected_alignment = np.array([(3,0),(4,1),(5,2),(6,4),(7,5),(9,6),(10,7),(11,8),(12,9),(13,10),(14,14),(19,15),(20,16),(21,17)]) 
    result = simpleAlign.align(unaligned_features_1, unaligned_features_2, lambda a:np.abs(a))
    np.testing.assert_almost_equal(result, expected_alignment)

def test_that_simple_align_aligns_more_dimensional_data_with_given_norm():
    simpleAlign = SimpleTimeAlign()
    unaligned_features_1 = np.array([1,2,3,4,5.1,6,7,8,9,10,11,12,13,14,15,15,15,15,15,16,17,18,18]) 
    unaligned_features_2 = np.array([4.1,5,6,6,7,8,10,11,12,13,14,14,14,14,15,16,17,18,19,20,21,22,23]) 
    unaligned_features_1 = np.stack([unaligned_features_1/2,unaligned_features_1/2] , axis=1)
    unaligned_features_2 = np.stack([unaligned_features_2/2,unaligned_features_2/2] , axis=1)
    expected_alignment = np.array([(3,0),(4,1),(5,2),(6,4),(7,5),(9,6),(10,7),(11,8),(12,9),(13,10),(14,14),(19,15),(20,16),(21,17)]) 
    result = simpleAlign.align(unaligned_features_1, unaligned_features_2, lambda a:np.abs(a[0])+np.abs(a[1]))
    np.testing.assert_almost_equal(result, expected_alignment)