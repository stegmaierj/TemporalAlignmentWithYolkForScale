#%%

import h5py
import numpy as np
import json
from PointCloudTransformer.TemporalShifter import TemporalShifter
from DataGenerator.SyntheticDataGenerator import SyntheticDataGenerator

#%%

#Configuration

#experiments: here you can instead configure your own data to use as base for synthetic ones.
#the h5 data needs to be in the following format:
#Pointcloud: x: x-coordinates of points (float)
#            y: y-coordinates of points (float)
#            z: z-coordinates of points (float)
#            time: time frame (int)
#            id: a unique id for a point
#            mother_id: the id of the same cell one time frame prior or of the mother cell in case of cell divison

# 

experiments = [{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_06.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_cos_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/cos_shift.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_06.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_gaussian_random.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/gaussian_random.txt',
},
{

    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_06.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_different_time_resolution.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/different_time_resolution.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_06.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_06_stable_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/stable_shift.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_08.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_cos_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/cos_shift.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_08.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_gaussian_random.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/gaussian_random.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_08.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_different_time_resolution.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/different_time_resolution.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_08.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_08_stable_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/stable_shift.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_11.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_cos_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/cos_shift.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_11.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_gaussian_random.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/gaussian_random.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_11.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_different_time_resolution.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/different_time_resolution.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_11.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_11_stable_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/stable_shift.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_12.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_cos_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/cos_shift.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_12.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_gaussian_random.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/gaussian_random.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_12.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_different_time_resolution.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/different_time_resolution.txt',
},
{
    'point_cloud': 'Data/Zebrafishembryos/embryo_ew_12.h5',
    'new_shifted_point_cloud': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryos/embryo_ew_12_stable_shift.h5',
    'temporal_alignment_file': 'Results/Evaluation/Experiments/TemporalShifts/stable_shift.txt',
}]
#REDO = False


for experiment in experiments:
    randomNumberGenerator = np.random.default_rng(12345)
    point_cloud = h5py.File(experiment['point_cloud'], 'r')
    new_shifted_point_cloud = h5py.File(experiment['new_shifted_point_cloud'], 'w')
    temporal_alignment_file = experiment['temporal_alignment_file']

    with open(temporal_alignment_file, 'r') as convert_file:
            temporalAlignment = np.array(json.load(convert_file)['time_transform'])

    time1 = np.array(point_cloud["Pointcloud"]["time"])
    x_1 = np.array(point_cloud["Pointcloud"]["x"])
    y_1 = np.array(point_cloud["Pointcloud"]["y"])
    z_1 = np.array(point_cloud["Pointcloud"]["z"])
    id = np.array(point_cloud["Pointcloud"]["id"])
    mother_id = np.array(point_cloud["Pointcloud"]["mother_id"])

    original_pc= { 'time': time1, 'x': x_1, 'y': y_1, 'z': z_1, 'id': id, 'mother_id': mother_id}

    #-------------------------------------------------------
    #------------------ Do temporal shift-------------------
    #-------------------------------------------------------

    shifter = TemporalShifter()
    #%%
    shifted_pc = shifter.shift(original_pc,temporalAlignment)

    #-------------------------------------------------------
    #------------------ Do scaling--------------------------
    #-------------------------------------------------------

    syntheticDataGenerator = SyntheticDataGenerator()

    shifted_pc, scale_factor = syntheticDataGenerator.scaleRandomly(inPlace=True, pc=shifted_pc, randomgenerator=randomNumberGenerator, scale_derivation=0.1, scale_mean=1)

    #-------------------------------------------------------
    #------------------ Do rotation--------------------------
    #-------------------------------------------------------

    shifted_pc, rotatation_angles = syntheticDataGenerator.rotateRandomly(randomgenerator=randomNumberGenerator, pc=shifted_pc, rotate_derivation=np.pi, rotate_mean=0, inPlace=True)

    #-------------------------------------------------------
    #------------------ Do translation----------------------
    #-------------------------------------------------------

    shifted_pc, translation_vector = syntheticDataGenerator.shiftRandomly(randomgenerator=randomNumberGenerator, pc=shifted_pc, shift_mean=0, shift_derivation=30, inPlace=True)

    
    #-------------------------------------------------------
    #------------------ Do scattering----------------------
    #-------------------------------------------------------

    shifted_pc = syntheticDataGenerator.scatterPoints(randomgenerator=randomNumberGenerator, pc=shifted_pc, noise_derivation=5, inPlace=True)

    #-------------------------------------------------------
    #------------------ Delete random points----------------
    #-------------------------------------------------------

    shifted_pc = syntheticDataGenerator.deletePoints(randomgenerator=randomNumberGenerator, pc=shifted_pc, prop_to_be_deleted=0.2, inPlace=True)

    #-------------------------------------------------------
    #------------------ Save synthetic data sample----------
    #-------------------------------------------------------


    group = new_shifted_point_cloud.create_group('Pointcloud')
    group.create_dataset('x', data=shifted_pc['x'])
    group.create_dataset('y', data=shifted_pc['y'])
    group.create_dataset('z', data=shifted_pc['z'])
    group.create_dataset('time', data=shifted_pc['time'])
    group.create_dataset('id', data=shifted_pc['id'])
    group.create_dataset('mother_id', data=shifted_pc['mother_id'])
    new_shifted_point_cloud.close()
