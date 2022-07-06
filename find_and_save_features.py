#%% Imports

import h5py
import numpy as np
from SpacialFeatures.Sphere import Sphere
from SpacialFeatures.SurroundingSphereFeature import SurroundingSphereFeature
from SpacialFeatures.SizeFeature import SizeFeature
import json
import os

#scripts that finds surrounding sphere and size feature for convigured embryos
#the h5 data needs to be in the following format:
#Pointcloud: x: x-coordinates of points (float)
#            y: y-coordinates of points (float)
#            z: z-coordinates of points (float)
#            time: time frame (int)
#            id: a unique id for a point
#            mother_id: the id of the same cell one time frame prior or of the mother cell in case of cell divison

TIMEFRAMES_MAX = 370

# you should replace this with your own experiments.
experiments = [{
    'point_cloud_file': 'Data/Zebrafishembryos/embryo_ew_08.h5',
    'feature_file': 'Results/FoundFeatures/features_08.txt'
}, {
    'point_cloud_file': 'Data/Zebrafishembryos/embryo_ew_06.h5',
    'feature_file': 'Results/FoundFeatures/features_06.txt'
},{
    'point_cloud_file': 'Data/Zebrafishembryos/embryo_ew_11.h5',
    'feature_file': 'Results/FoundFeatures/features_11.txt'
},{
    'point_cloud_file': 'Data/Zebrafishembryos/embryo_ew_12.h5',
    'feature_file': 'Results/FoundFeatures/features_12.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_cos_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_cos_shift.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_gaussian_random.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_gaussian_random.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_different_time_resolution.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_different_time_resolution.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_stable_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_stable_shift.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_cos_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_cos_shift.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_gaussian_random.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_gaussian_random.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_different_time_resolution.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_different_time_resolution.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_stable_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_stable_shift.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_cos_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_cos_shift.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_gaussian_random.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_gaussian_random.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_different_time_resolution.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_different_time_resolution.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_stable_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_stable_shift.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_cos_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_cos_shift.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_gaussian_random.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_gaussian_random.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_different_time_resolution.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_different_time_resolution.txt'
},{
    'point_cloud_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_stable_shift.h5',
    'feature_file': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_stable_shift.txt'
}]
REDO = False

#%% feature calculation
def calc_features(pointCloud):
    
    surrSphereFeature = SurroundingSphereFeature(type='optimization')
    sizeFeature = SizeFeature()

    time = np.array(pointCloud["Pointcloud"]["time"])
    pc = {
        'x': pointCloud["Pointcloud"]["x"][time==min(time)],
        'y': pointCloud["Pointcloud"]["y"][time==min(time)],
        'z': pointCloud["Pointcloud"]["z"][time==min(time)]
    }
    sphere = surrSphereFeature.apply(pc)

    size = np.full(TIMEFRAMES_MAX,-1, dtype=float)
    number_of_points = np.full(TIMEFRAMES_MAX,-1, dtype=int)
    for i in range(TIMEFRAMES_MAX):
        pc = {
        'x': pointCloud["Pointcloud"]["x"][time==(i+1)],
        'y': pointCloud["Pointcloud"]["y"][time==(i+1)],
        'z': pointCloud["Pointcloud"]["z"][time==(i+1)]
        }
        number_of_points[i] = pc['x'].size
        # time frame does not exist:
        if number_of_points[i] == 0:
            size[i] = 0
        else:
            size[i] = sizeFeature.apply(pc, sphere)
        print('{}/{}} done'.format(i, TIMEFRAMES_MAX))
    return (sphere, size,number_of_points)

def save_features(name, sphere: Sphere, size, number_of_points):
    #%% Covert to json and save:

    sphere_json = sphere.dir_representation()
    features_json = {'sphere': sphere_json,
                    'size':  size.tolist(),
                    'number_of_points': number_of_points.tolist()}
    with open(name, 'w') as convert_file:
        convert_file.write(json.dumps(features_json))

for experiment in experiments:
    if ((REDO == False) & (os.path.exists(experiment['feature_file']))):
        continue
    point_cloud = h5py.File(experiment['point_cloud_file'], 'r')
    sphere, size, number_of_points = calc_features(point_cloud)
    save_features(experiment['feature_file'], sphere, size, number_of_points)
    print('saved features {}'.format(experiment['feature_file']))

        



# %%
# %%
