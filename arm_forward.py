import numpy as np
import cupy as cp

# input_feat = np.load("./sample_input/0/features_1001423e-20220630222223.655.npy")
input_feat = np.load("./sample_input/1/features_1001423e-20220628162912.729.npy")
# input_feat = np.load("/home/uncanny/projects/CUDA/backbone_task/train_data_for_classification/1/features_1001423e-20220718104538.023.npy")

# print(np.allclose(input_feat, input_feat_1))


weight_1 = np.load("./weights_classification/fully_connected_network1.weight.npy")
bias_1 = np.load("./weights_classification/fully_connected_network1.bias.npy")

weight_2 = np.load("./weights_classification/fully_connected_network2.weight.npy")
bias_2 = np.load("./weights_classification/fully_connected_network2.bias.npy")


print("Input shape : ", input_feat.shape)

print("Weights shape : ", weight_1.shape) 
print("Bias shape : ", bias_1.shape)

print("Weights shape : ", weight_2.shape)
print("Bias shape : ", bias_2.shape)

# Normal
def Sigmoid_cupy(x):
    return 1/(1+cp.exp(-x))

# Stable version
def Sigmoid_stable_cupy(x):
    return cp.exp(-cp.logaddexp(0., -x)) 


with cp.cuda.Device(0):
    print("On GPU area")
    input_gpu = cp.asarray(input_feat)
    weight_1_gpu = cp.asarray(weight_1)
    bias_1_gpu = cp.asarray(bias_1)
    
    weight_2_gpu = cp.asarray(weight_2)
    bias_2_gpu = cp.asarray(bias_2)

    out_1_gpu = cp.add(cp.matmul(input_gpu, cp.transpose(weight_1_gpu)), bias_1_gpu)
    print(out_1_gpu[:10])

    out_2_gpu = cp.add(cp.matmul(out_1_gpu, cp.transpose(weight_2_gpu)), bias_2_gpu) 
    
    print(out_2_gpu)

    
    
