# create proj and set fpga target
open_project -reset lenet5
set_top lenet5

# lib
add_files ../src/lenet5.cpp
add_files ../src/conv2d.cpp
add_files ../src/avg_pool.cpp
add_files ../src/flatten.cpp
add_files ../src/fc_layer.cpp

# include
add_files ../include/lenet5.h
add_files ../include/conv2d.h
add_files ../include/avg_pool.h
add_files ../include/flatten.h
add_files ../include/fc_layer.h

# tb
add_files -tb ../test/test_utils.h
add_files -tb ../test/test_conv2d.cpp
add_files -tb ../test/test_avg_pool.cpp
add_files -tb ../test/test_fc_layer.cpp
add_files -tb ../test/test_lenet5.cpp

config_interface -m_axi_addr64
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_auto_max_ports

# c sim
csim_design -argv "conv2d"   # Test conv layer
# csim_design -argv "avg_pool"   # Test avg pool layer
# csim_design -argv "fc_layer"   # Test fc layer
# csim_design -argv "lenet"   # Test full model

csynth_design

cosim_design -rtl verilog -trace_level all

export_design -format ip_catalog
