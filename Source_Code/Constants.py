ROOT_DIR = "../"
MODEL_WEIGHTS_SUBDIR = "./Model_Weights/"

# .h5 Keras Weight File Storage Directories:

## Mutliple Output Fully Conected Neural Network
H5_DIR_FULLYCONNECTED_MULTIPLE_RELU = MODEL_WEIGHTS_SUBDIR + "fullyconnected_multiple.h5"
H5_DIR_FULLYCONNECTED_MULTIPLE_ELU = MODEL_WEIGHTS_SUBDIR + "fullyconnected_multiple_elu.h5"

## Multiple Output Convolutional Networks
H5_DIR_COVNET_MULTIPLE_RELU = MODEL_WEIGHTS_SUBDIR + "covnet_multiple_running.h5"
H5_DIR_COVNET_MULTIPLE_ELU = MODEL_WEIGHTS_SUBDIR + "covnet_multiple_elu.h5"

## Single Output Convolution Networks
H5_DIR_COVNET_JUMP_SIGMA = MODEL_WEIGHTS_SUBDIR + "covnet_jump_sigma_running.h5"
H5_DIR_COVNET_JUMP_MU = MODEL_WEIGHTS_SUBDIR + "covnet_jump_mu_running.h5"
H5_DIR_COVNET_LAMBDA = MODEL_WEIGHTS_SUBDIR + "covnet_lambda_running.h5"

## Ensembling Network
H5_DIR_ENSEMBLING = MODEL_WEIGHTS_SUBDIR + "ff_ensembling_large.h5"


# History Files
HISTORY_DIR_FEEDFORWARD_RELU = MODEL_WEIGHTS_SUBDIR + "feedforward_relu_histories.npy"


# Prediction History Files
PREDICTIONS_DIR_FEEDFORWARD_RELU = MODEL_WEIGHTS_SUBDIR + "feedforward_relu_predictions.npy"
