open_project lenet5_hls
set_top lenet5_top

add_files data/weights_bias.h
add_files kernels/aes_utils.h
add_files kernels/utils.h
add_files kernels/conv2d.h
add_files kernels/avg_pool.h
add_files kernels/flatten.h
add_files kernels/fc.h
add_files kernels/lenet5_top.cpp

open_solution sol1 -flow_target vitis

set_part  xcvu37p-fsvh2892-2L-e
create_clock -period 3.0 -name default
config_compile -pipeline_loops 1     

csynth_design
export_design \
    -rtl verilog \
    -format ip_catalog \
    -output lenet5_ip \
    -ipname lenet5_kernel
exit
