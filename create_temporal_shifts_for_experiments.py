#%%
import numpy as np
import json
import matplotlib.pyplot as plt


#With this script the random shift functions are created that are used to create the synthetic embryos:

# If you want to create temporal shifts for your own data you have to adopt the frame number acordingly.
#%%

#configure wether you want to visulize the features as plots in the end
VISUALIZE = False
FRAME_NUMBER = 370
def generate_shift_map(shift_beginning, shift_function, ):
    pairs = np.full(shape=FRAME_NUMBER, fill_value = -1, dtype=int)

    pair_0 = 0
    pair_1 = 0

    if shift_beginning < 0:
        pair_0 = 0
        pair_1 = - int(shift_beginning)
    elif shift_beginning > 0:
        pair_0 = int(shift_beginning)
        pair_1 = 0
    else:
        pair_0 = 0
        pair_1 = 0
    while (pair_0 < FRAME_NUMBER) & (pair_1 <FRAME_NUMBER):
        pairs[pair_0] = pair_1
        new_shift = shift_function(pair_0, pair_1)
        if new_shift < 0:
            pair_0 = pair_0 + 1
            pair_1 = pair_1 + 1 - int(new_shift)
        elif new_shift > 0:
            pair_0 = pair_0 + 1 + int(new_shift)
            pair_1 = pair_1 + 1
        else:
            pair_0 = pair_0 + 1
            pair_1 = pair_1 + 1


    return np.stack([np.where(pairs>-1)[0], pairs[pairs>-1]], axis=1)

#%%
#1.1 Random gaussian shifts

radom_number_generator = np.random.default_rng(12345)

mean_shift = 0
std_shift_beginning = 40
std_shift_from_linear = 10
temporal_alignment_random = generate_shift_map(radom_number_generator.normal(mean_shift, std_shift_beginning, size=1),
    lambda i,j: radom_number_generator.normal(mean_shift, std_shift_from_linear, size=1))
with open('Results/Evaluation/Experiments/TemporalShifts/gaussian_random.txt', 'w') as convert_file:
    convert_file.write(json.dumps({'time_transform': temporal_alignment_random.tolist()}))

#---------------------------------------------------------------------

# %%
#1.2 stable shift with some gaussian noise
radom_number_generator = np.random.default_rng(12345)
mean_shift = 0
std_shift_beginning = 40
std_shift_from_linear = 1
temporal_alignment_stable = generate_shift_map(radom_number_generator.normal(mean_shift, std_shift_beginning, size=1),
    lambda i,j: radom_number_generator.normal(mean_shift, std_shift_from_linear, size=1))
with open('Results/Evaluation/Experiments/TemporalShifts/stable_shift.txt', 'w') as convert_file:
    convert_file.write(json.dumps({'time_transform': temporal_alignment_stable.tolist()}))
#---------------------------------------------------------------------

# %%
#1.3 different resolution in time

shift_beginning = -10
scaling_factor = 3

def shift_function_scaling(i,j):
    return  scaling_factor

temporal_alignment_diff_res = generate_shift_map(shift_beginning, shift_function_scaling)
with open('Results/Evaluation/Experiments/TemporalShifts/different_time_resolution.txt', 'w') as convert_file:
    convert_file.write(json.dumps({'time_transform': temporal_alignment_diff_res.tolist()}))
# %%
#---------------------------------------------------------------------

#1.4 embryo 1 is slower in first half, and faster in second half (cos)
shift_beginning = 10

def shift_function_cos(i,j):
    return np.cos(i/100*np.pi)*2

temporal_alignment_cos = generate_shift_map(shift_beginning, shift_function_cos)
with open('Results/Evaluation/Experiments/TemporalShifts/cos_shift.txt', 'w') as convert_file:
    convert_file.write(json.dumps({'time_transform': temporal_alignment_cos.tolist()}))
#---------------------------------------------------------------------

#%%
#1.5 embryo 1 is faster in first half and slower in second half (sin)

shift_beginning = -10

def shift_function_sin(i,j):
    return np.sin(i/100*np.pi)*2

temporal_alignment_sin = generate_shift_map(shift_beginning, shift_function_sin)
with open('Results/Evaluation/Experiments/TemporalShifts/sin_shift.txt', 'w') as convert_file:
    convert_file.write(json.dumps({'time_transform': temporal_alignment_sin.tolist()}))
#---------------------------------------------------------------------

# %%
if VISUALIZE:
#visulaize

    fig = plt.subplots(figsize =(10, 7))

    plt.plot(temporal_alignment_random[:,0], temporal_alignment_random[:,1], label='random transform')
    plt.plot(temporal_alignment_stable[:,0], temporal_alignment_stable[:,1], label='stable transform')

    plt.plot(temporal_alignment_diff_res[:,0], temporal_alignment_diff_res[:,1], label='different resolution')
    plt.plot(temporal_alignment_cos[:,0], temporal_alignment_cos[:,1], label='cosinus transform')
    plt.plot(temporal_alignment_sin[:,0], temporal_alignment_sin[:,1], label='sinus transform')


    plt.ylabel('Time Embryo 1')
    plt.xlabel('Time Embryo 2')
    plt.title('Temporal Transforms')

    #plt.xticks(ind, ('0', '50', '100', '150', '200'))
    plt.legend()

    plt.show()

# %%
