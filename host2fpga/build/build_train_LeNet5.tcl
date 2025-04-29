open_project train_lenet5_hls
set_top train_lenet5_top

add_files data/weights_bias_raw_params.h
add_files kernels/aes_utils.h
add_files kernels/utils.h
add_files kernel/reader.h

add_files kernels/conv2d.h
add_files kernels/conv2d_bwd.h
add_files kernels/avg_pool.h
add_files kernels/avg_pool_bwd.h
add_files kernels/flatten.h
add_files kernels/flatten_bwd.h
add_files kernels/fc.h
add_files kernels/fc_bwd.h

add_files kernels/update.h
add_files kernels/mse_loss.h

add_files kernels/train.h
add_files kernels/train.cpp

add_files -tb test/test_train.cpp -cflags "-I./kernels"

open_solution sol1 -flow_target vitis

set_part  xcvu37p-fsvh2892-2L-e
create_clock -period 3.0 -name default
config_compile -pipeline_loops 1     

# csim_design
csynth_design
cosim_design
export_design \
    -rtl verilog \
    -format ip_catalog \
    -output lenet5_ip \
    -ipname lenet5_kernel
exit
