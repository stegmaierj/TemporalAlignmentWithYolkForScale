#%%
from Alignment.SimpleTimeAlign import SimpleTimeAlign
import json
import numpy as np

#norm features:

#%%

#You can use your own data here instead. Use features created with find_and_save_features_script


# experiments for evaluation of registration:
'''
experiments = [{
        'features_1': 'Results/FoundFeatures/features_06.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_cos_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_06_too_ew_06_cos_shift.txt'

},{
        'features_1': 'Results/FoundFeatures/features_06.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_different_time_resolution.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_06_too_ew_06_different_time_resolution.txt'
},{
        'features_1': 'Results/FoundFeatures/features_06.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_gaussian_random.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_06_too_ew_06_gaussian_random.txt'
},{
        'features_1': 'Results/FoundFeatures/features_06.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_06_stable_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_06_too_ew_06_stable_shift.txt'
},{
        'features_1': 'Results/FoundFeatures/features_08.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_cos_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_08_too_ew_08_cos_shift.txt'

},{
        'features_1': 'Results/FoundFeatures/features_08.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_different_time_resolution.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_08_too_ew_08_different_time_resolution.txt'
},{
        'features_1': 'Results/FoundFeatures/features_08.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_gaussian_random.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_08_too_ew_08_gaussian_random.txt'
},{
        'features_1': 'Results/FoundFeatures/features_08.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_08_stable_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_08_too_ew_08_stable_shift.txt'
},{
        'features_1': 'Results/FoundFeatures/features_11.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_cos_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_11_too_ew_11_cos_shift.txt'

},{
        'features_1': 'Results/FoundFeatures/features_11.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_different_time_resolution.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_11_too_ew_11_different_time_resolution.txt'
},{
        'features_1': 'Results/FoundFeatures/features_11.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_gaussian_random.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_11_too_ew_11_gaussian_random.txt'
},{
        'features_1': 'Results/FoundFeatures/features_11.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_11_stable_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_11_too_ew_11_stable_shift.txt'
},{
        'features_1': 'Results/FoundFeatures/features_12.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_cos_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_12_too_ew_12_cos_shift.txt'

},{
        'features_1': 'Results/FoundFeatures/features_12.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_different_time_resolution.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_12_too_ew_12_different_time_resolution.txt'
},{
        'features_1': 'Results/FoundFeatures/features_12.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_gaussian_random.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_12_too_ew_12_gaussian_random.txt'
},{
        'features_1': 'Results/FoundFeatures/features_12.txt',
        'features_2': 'Results/Evaluation/Experiments/ShiftedZebrafishEmbryoFeatures/embryo_ew_12_stable_shift.txt',
        'alignment_file': 'Results/Evaluation/Experiments/ExperimentResults/RegistrationResults/ew_12_too_ew_12_stable_shift.txt'
}]
'''
# experiments for real registration:
experiments = [{
        'features_1': 'Results/FoundFeatures/features_06.txt',
        'features_2': 'Results/FoundFeatures/features_08.txt',
        'alignment_file': 'Results/TemporalRegistration/register_06_to_08.txt'
},
{
        'features_1': 'Results/FoundFeatures/features_06.txt',
        'features_2': 'Results/FoundFeatures/features_11.txt',
        'alignment_file': 'Results/TemporalRegistration/register_06_to_11.txt'
},{
        'features_1': 'Results/FoundFeatures/features_06.txt',
        'features_2': 'Results/FoundFeatures/features_12.txt',
        'alignment_file': 'Results/TemporalRegistration/register_06_to_12.txt'
}]

registrator = SimpleTimeAlign()

for experiment in experiments:

        with open(experiment['features_1'], 'r') as convert_file:
                features_1 = json.load(convert_file)
        with open(experiment['features_2'], 'r') as convert_file:
                features_2 = json.load(convert_file)


        #norm sizes with radius to be comparable:
        radius_1 = features_1['sphere']['radius']
        radius_2 = features_2['sphere']['radius']
        size_1 = np.array(features_1['size'])/radius_1
        size_2 = np.array(features_2['size'])/radius_2
        alignment = registrator.align(size_1, size_2, lambda a:np.abs(a))
        #save results:
        with open(experiment['alignment_file'], 'w') as convert_file:
                convert_file.write(json.dumps({'time_transform': alignment.tolist()}))

