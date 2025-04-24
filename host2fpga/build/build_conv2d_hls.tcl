open_project conv2d_hls
set_top conv2d_kernel

add_files kernels/conv2d_kernel.cpp
add_files kernels/aes_utils.h

open_solution sol1 -flow_target vitis

set_part  xcvu37p-fsvh2892-2L-e
create_clock -period 3.0 -name default
config_compile -pipeline_loops 1     

csynth_design
export_design \
    -rtl verilog \
    -format ip_catalog \
    -output conv2d_ip \
    -ipname conv2d_kernel
exit
