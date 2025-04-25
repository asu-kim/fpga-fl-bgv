open_project avg_pool_hls
set_top avg_pool_kernel

add_files kernels/avg_pool_kernel.cpp
add_files kernels/aes_utils.h

open_solution sol1 -flow_target vitis

set_part  xcvu37p-fsvh2892-2L-e
create_clock -period 3.0 -name default
config_compile -pipeline_loops 1     

csynth_design
export_design \
    -rtl verilog \
    -format ip_catalog \
    -output avg_pool_ip \
    -ipname avg_pool_kernel
exit
