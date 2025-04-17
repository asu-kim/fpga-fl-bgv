# FPGA accelerated Client-Side of Federated Learning with BGV Homomorphic Encryption Algorithm
## Prerequisite
```
Vivado
Vitis
Vitis-HLS
Xilinx XRT
```

## Build and Test
1. Test the functionaility and build xo module.
```
cd build/
vitis_hls run_hls.tcl
```
Modify `set_top` and `set hls_exec` appropriate to the purpose.
`hls_exec 1` or `hls_exec 2` only test the functionality and `hls_exec 3` will generate *.xo file.

2. Build `xclbin` file.
```
v++ -l -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --kernel {top_module} {top_module}.xo -o {top_module}.xclbin
```

3. Sign `xclbin` file
```
sudo /opt/xilinx/xrt/bin/xclbinutil --private-key /var/lib/shim-signed/mok/MOK.priv --certificate /var/lib/shim-signed/mok/MOK.der --input {top_module}.xclbin --output {top_module}_signed.xclbin
```

4. Compile host.cpp
At the top directory, run `make host`.

# Note
to build and simulate LeNet5, change the project root directory in test/test_utils.h
