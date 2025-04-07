# Install Vitis-AI
## ML framework: PyTorch

1. Train your model in pytorch
2. Store the scripted model as a .ptr file
3. Use docker env provided by Vitis-AI to run the script: vitis_ai_torch_quantizer.py

You will get a quant_result_test folder with 2 subfolders 
1. compiled_model: this contains a .xmodel 
2. deploy_check_data_int: this contains the binaries for your quantized weights, biases, scaling factors and zeropoints
