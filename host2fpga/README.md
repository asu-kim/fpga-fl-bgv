## Build
build scripts
## Driver
driver code called by build
code here loads weight from data/weights_bias.h
pacakge params together and send to lenet module
## kernels
- conv2d.h (logic)
- conv2d_kernel.cpp (read data from HBM and communicate with logic)
- avg_pool.h
- avg_pool_kernel.cpp
- flatten.h
- flatten_kernel.cpp
- fc.h
- fc_kernel.cpp
- lenet5_top.cpp (read data from HBM and called logic)
## quant data
contains updated weights in text format
## scripts
contains code to generate data/weights_bias from quant_data
