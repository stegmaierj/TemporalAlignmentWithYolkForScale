# TemporalAlignmentWithYolkForScale


This is the repository for the paper "Temporal Registration of Early Zebrafish Embryos Using Biological Prior Knowledge". You can use this repository to reproduce all experiments and evaluations and to follow a Python implementation of the approach described in the paper. Please cite the paper when using our work.

## Installation

Install the package with 

_pip install -r requirements.txt_


## Usage

### Redoing the Experiments from the Paper:

There are scripts for conducting the experiments from the paper in the main folder, also for creating the figures, finding the features (yolk approximation and size) and registration based on these features. For redoing the experiments you still need the original data, please contact us via email (ina.laube@lfb.rwth-aachen.de or johannes.stegmaier@lfb.rwth-aachen.de) to get the zebrafish embryos (original from [1]) in the used file format.

* The sphere evaluation experiments meantioned in the paper can be redone with the scripts evaluate_sphere_features_exp1 and evaluate_sphere_feature_exp2
* The synthetic data can be created with the create_data_with_known_shifts script
* The registration of the synthetic data to the reference ones can be done by first creating the feature files of the synthtic data samples with find_and_save_features and then register with the temporal_registration_using_relative_size script
### Using the algorithmn on your own data:

You can use and modify our code in your own project, just make sure to cite our work in case of publication. Since the number of frames and the realistic range of yolk sizes are hard-coded, you will probably need to make some adjustments to the code to apply it to your own data. If you run into problems doing this, you can always contact us by email.

The scripts should already give a good impression on how to use the code, the pipeline is:

#### Creating Synthetic data:
* create_temporal_shifts_for_experiments.py contains the generation of the 4 used temporal shifts and can also be used as an orientation if you want to use your own shift functions
* create_data_with_known_shifts.py contains the generation of the synthetic data samples using reference embryos and shift functions. 

#### Registration point clouds temporal:
* find_and_save_features.py is a script that shows you how to use the code to find the surrounding sphere (yolk approximation) and size features
* temporal_registration_using_relative_size shows you how to use the found features for registration

At the moment, there is no ready-made version that you can simply install via pip and use within your own project. It is recommended to fork this git repository and modify it for your own needs.

## Support

Write ina.laube@lfb.rwth-aachen.de or johannes.stegmaier@lfb.rwth-aachen.de or create a GitHub Issue if you need any help / run into any issues.

## Authors and acknowledgment

Programming done by Ina Laube. Research done by Ina Laube, Dennis Eschweiler, Abin Jose and Johannes Stegmaier. We thank the German Research Foundation DFG for funding (IL, Grant No STE2802/1-1, DE, Grant No STE2802/2-1 ).

The zebrafish data used is from the paper:

[1] Kobitski, Andrei Y. et al.: An Ensemble-Averaged, Cell Density-based Digital Model
of Zebrafish Embryo Development Derived from Light-Sheet Microscopy Data with
Single-Cell Resolution. Scientific Reports 5(1), 8601 (2015)

## License

Apache 2.0 (see license file) 

