#%%

import h5py
import json
import numpy as np
from SpacialFeatures.Sphere import Sphere

from SpacialFeatures.SurroundingSphereFeature import SurroundingSphereFeature
from DataGenerator.SyntheticDataGenerator import SyntheticDataGenerator

#script for sphere evaluation experiment 2: calculate sphere on synthetic samples and see variance


data_folder = "Data/Zebrafishembryos"
embryos = ["ew_12"]
redo = False
syntheticDataGenerator = SyntheticDataGenerator()
surrSphereFeature = SurroundingSphereFeature(type='optimization')

def evaluate_with_random_scale_and_shift(sphere_1, randomNumberGenerator, pc):

    #create synthetic sample with random scale and translate:
    shifted_pc, scale_factor, translate = syntheticDataGenerator.createSyntheticSampleWithTranslateAndScale(randomNumberGenerator,pc,sphere_1.middlepoint,[0,30],[1,0.1])

    #calculate sphere on synthetic sample:
    sphere_for_shifted = surrSphereFeature.apply(shifted_pc)

    middle_point_loss = np.linalg.norm(sphere_for_shifted.middlepoint - np.array([translate[0], translate[1], translate[2]]))
    radius_loss = np.abs(sphere_for_shifted.radius - np.array(scale_factor*sphere_1.radius))
    return (middle_point_loss, radius_loss, scale_factor, translate[0], translate[1], translate[2], sphere_for_shifted.middlepoint[0], sphere_for_shifted.middlepoint[1], sphere_for_shifted.middlepoint[2], sphere_for_shifted.radius)

for embryo in embryos:
    randomNumberGenerator = np.random.default_rng(12345)

    point_cloud = h5py.File('{}/embryo_{}.h5'.format(data_folder, embryo), 'r')['Pointcloud']

    time = np.array(point_cloud["time"])

    with open('sphereEval/{}/sphere_eval_1.txt'.format(embryo), 'r') as convert_file:
        sphere_1 = Sphere.fromDict(json.load(convert_file))

    for i in np.arange(50):
        (middle_point_loss, radius_loss, scale_factor, translateX, translateY, translateZ, m_o, m_1, m_2, r) = evaluate_with_random_scale_and_shift(sphere_1, randomNumberGenerator, {'x': np.array(point_cloud['x'])[time==1], 'y': np.array(point_cloud['y'])[time==1], 'z': np.array(point_cloud['z'])[time==1]})
        with open('sphereEval/{}/eval_sphere_exp2_add_inf.csv'.format(embryo), 'a') as convert_file:
            convert_file.write("{},{}, {},{}, {},{},{},{},{},{}\n".format(middle_point_loss, radius_loss, scale_factor, translateX, translateY, translateZ, m_o, m_1, m_2, r))
