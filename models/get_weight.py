import os
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="path to the model to get weights")
args = parser.parse_args()

out_dir = "quant_params_in_int" 
os.makedirs(out_dir, exist_ok=True)

state_dict = torch.load(args.model)

def save_array_as_text(array, file_path):
    with open(file_path, "w") as f:
        f.write(f"# Shape: {array.shape}\n")
        f.write("# C-style row major order\n")
        if array.ndim > 2:
            array = array.flatten()
        arr_to_save = array.reshape(1, -1) if array.ndim == 1 else array
        np.savetxt(f, arr_to_save, fmt='%d', delimiter=', ')
    print(f"Saved to {file_path}")

# CONV1 weight
conv1_weight = state_dict["conv1.weight"].cpu().numpy()
conv1_weight_scale = state_dict["conv1.weight_fake_quant.scale"].cpu().numpy()
conv1_weight_zp = state_dict["conv1.weight_fake_quant.zero_point"].cpu().numpy()

# oc, ic, h, w
for oc in range(conv1_weight.shape[0]):
    conv1_weight[oc] = np.round((conv1_weight[oc] - conv1_weight_zp[oc]) / conv1_weight_scale[oc])

# COVN1 bias
conv1_bias = state_dict["conv1.bias"].cpu().numpy()
conv1_bias_post_scale = state_dict["conv1.activation_post_process.scale"].item()

channels = conv1_bias.shape[0]
conv1_bias_int8 = np.zeros(channels, dtype=np.int8)
for oc in range(channels):
    bias_float_val = conv1_bias[oc]
    bias_scaled = bias_float_val / (conv1_weight_scale[oc] * conv1_bias_post_scale)
    conv1_bias_int8[oc] = int(round(bias_scaled))

# CONV2
conv2_weight = state_dict["conv2.weight"].cpu().numpy()
conv2_weight_scale = state_dict["conv2.weight_fake_quant.scale"].cpu().numpy()
conv2_weight_zp = state_dict["conv2.weight_fake_quant.zero_point"].cpu().numpy()

# oc, ic, h, w
for oc in range(conv2_weight.shape[0]):
    conv2_weight[oc] = np.round((conv2_weight[oc] - conv2_weight_zp[oc]) / conv2_weight_scale[oc])

# COVN2 bias
conv2_bias = state_dict["conv2.bias"].cpu().numpy()
conv2_bias_post_scale = state_dict["conv2.activation_post_process.scale"].item()

channels = conv2_bias.shape[0]
conv2_bias_int8 = np.zeros(channels, dtype=np.int8)
for oc in range(channels):
    bias_float_val = conv2_bias[oc]
    bias_scaled = bias_float_val / (conv2_weight_scale[oc] * conv2_bias_post_scale)
    conv2_bias_int8[oc] = int(round(bias_scaled))

save_array_as_text(conv1_weight, os.path.join(out_dir, "conv1_weight_int8.txt"))
save_array_as_text(conv2_weight, os.path.join(out_dir, "conv2_weight_int8.txt"))
save_array_as_text(conv1_bias_int8, os.path.join(out_dir, "conv1_bias_int8.txt"))
save_array_as_text(conv2_bias_int8, os.path.join(out_dir, "conv2_bias_int8.txt"))

# FC1
fc1_weight = state_dict["fc1.weight"].cpu().numpy()
fc1_weight_scale = state_dict["fc1.weight_fake_quant.scale"].cpu().numpy()
fc1_weight_zp = state_dict["fc1.weight_fake_quant.zero_point"].cpu().numpy()

# weight: all ele in one row quant with sqame param
for row in range(fc1_weight.shape[0]):
    fc1_weight[row, :] = np.round((fc1_weight[row, :] - fc1_weight_zp[row]) / fc1_weight_scale[row])
    fc1_weight[row, :] = np.clip(fc1_weight[row, :], -128, 127)

# FC1 bias
fc1_bias = state_dict["fc1.bias"].cpu().numpy()
fc1_bias_post_scale = state_dict["fc1.activation_post_process.scale"].item()

out_dim = fc1_bias.shape[0]
fc1_bias_int8 = np.zeros(out_dim, dtype=np.int8)

for oc in range(out_dim):
    bias_float_val = fc1_bias[oc]
    bias_scaled = bias_float_val / (fc1_weight_scale[oc] * fc1_bias_post_scale)
    fc1_bias_int8[oc] = int(round(bias_scaled))

# FC2
fc2_weight = state_dict["fc2.weight"].cpu().numpy()
fc2_weight_scale = state_dict["fc2.weight_fake_quant.scale"].cpu().numpy()
fc2_weight_zp = state_dict["fc2.weight_fake_quant.zero_point"].cpu().numpy()

# weight: all ele in one row quant with sqame param
for row in range(fc2_weight.shape[0]):
    fc2_weight[row, :] = np.round((fc2_weight[row, :] - fc2_weight_zp[row]) / fc2_weight_scale[row])
    fc2_weight[row, :] = np.clip(fc2_weight[row, :], -128, 127)

# FC2 bias
fc2_bias = state_dict["fc2.bias"].cpu().numpy()
fc2_bias_post_scale = state_dict["fc2.activation_post_process.scale"].item()

out_dim = fc2_bias.shape[0]
fc2_bias_int8 = np.zeros(out_dim, dtype=np.int8)

for oc in range(out_dim):
    bias_float_val = fc2_bias[oc]
    bias_scaled = bias_float_val / (fc2_weight_scale[oc] * fc2_bias_post_scale)
    fc2_bias_int8[oc] = int(round(bias_scaled))

# FC3
fc3_weight = state_dict["fc3.weight"].cpu().numpy()
fc3_weight_scale = state_dict["fc3.weight_fake_quant.scale"].cpu().numpy()
fc3_weight_zp = state_dict["fc3.weight_fake_quant.zero_point"].cpu().numpy()

# weight: all ele in one row quant with sqame param
for row in range(fc3_weight.shape[0]):
    fc3_weight[row, :] = np.round((fc3_weight[row, :] - fc3_weight_zp[row]) / fc3_weight_scale[row])
    fc3_weight[row, :] = np.clip(fc3_weight[row, :], -128, 127)

# FC3 bias
fc3_bias = state_dict["fc3.bias"].cpu().numpy()
fc3_bias_post_scale = state_dict["fc3.activation_post_process.scale"].item()

out_dim = fc3_bias.shape[0]
fc3_bias_int8 = np.zeros(out_dim, dtype=np.int8)

for oc in range(out_dim):
    bias_float_val = fc3_bias[oc]
    bias_scaled = bias_float_val / (fc3_weight_scale[oc] * fc3_bias_post_scale)
    fc3_bias_int8[oc] = int(round(bias_scaled))

save_array_as_text(fc1_weight, os.path.join(out_dir, "fc1_weight_int8.txt"))
save_array_as_text(fc2_weight, os.path.join(out_dir, "fc2_weight_int8.txt"))
save_array_as_text(fc3_weight, os.path.join(out_dir, "fc3_weight_int8.txt"))

save_array_as_text(fc1_bias_int8, os.path.join(out_dir, "fc1_bias_int8.txt"))
save_array_as_text(fc2_bias_int8, os.path.join(out_dir, "fc2_bias_int8.txt"))
save_array_as_text(fc3_bias_int8, os.path.join(out_dir, "fc3_bias_int8.txt"))
