open_project -reset compiled/lenet5
set_top lenet5

# include
add_files ../include/lenet5.h
add_files ../include/conv2d.h
add_files ../include/avg_pool.h
add_files ../include/flatten.h
add_files ../include/fc_layer.h
add_files ../test/test_utils.h

# tb
add_files -tb ../test/test_lenet5.cpp -cflags "-I../include -I../test"

open_solution -reset solution1 -flow_target vivado

# Define technology and clock rate
set_part  {xcu55c-fsvh2892-2L-e}
create_clock -period 10

config_interface -m_axi_addr64
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_auto_max_ports

# c sim

# Test conv2d layer
csim_design 

# csynth_design
# 
# cosim_design -rtl verilog -trace_level all
# 
# export_design -format ip_catalog
