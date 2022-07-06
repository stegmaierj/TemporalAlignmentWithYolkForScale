#%%

import h5py
import json
import numpy as np
import os

from SpacialFeatures.SurroundingSphereFeature import SurroundingSphereFeature
data_folder = "Data/Zebrafishembryos"
embryos = ["ew_06", "ew_08", "ew_11", "ew_12"]
redo = False

#script for sphere evaluation experiment 1: calculate sphere on first 50 time points and see variance

for embryo in embryos:
    point_cloud = h5py.File('{}/embryo_{}.h5'.format(data_folder, embryo), 'r')['Pointcloud']

    surrSphereFeature = SurroundingSphereFeature(type='optimization')

    time = np.array(point_cloud["time"])

    timepoints = np.arange(1, 40)
    if not os.path.exists('sphereEval/{}'.format(embryo)):
        os.makedirs('sphereEval/{}'.format(embryo))
    for t in timepoints: 
        pc_t = {
        'x': point_cloud["x"][time==t],
        'y': point_cloud["y"][time==t],
        'z': point_cloud["z"][time==t]
        }   
        if os.path.isfile('sphereEval/{}/sphere_eval_{}.txt'.format(embryo,t)):
            continue
        sphere_1 = surrSphereFeature.apply(pc_t)
        sphere_json = sphere_1.dir_representation()
        with open('sphereEval/{}/sphere_eval_{}.txt'.format(embryo,t), 'w') as convert_file:
            convert_file.write(json.dumps(sphere_json))

    print('done with {}'.format(embryo))
