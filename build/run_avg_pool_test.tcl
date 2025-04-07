open_project -reset compiled/test_avg_pool
set_top avg_pool

# lib
add_files ../include/avg_pool.h
add_files ../test/test_utils.h

# tb
add_files -tb ../test/test_avg_pool.cpp -cflags "-I../include -I../test"

open_solution -reset solution1 -flow_target vivado

# Define technology and clock rate
set_part  {xcu55c-fsvh2892-2L-e}
create_clock -period 10

config_interface -m_axi_addr64
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_auto_max_ports

# c sim
csim_design
