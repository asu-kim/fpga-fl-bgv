PLATFORM := blanche
TARGET   := hw   # or hw_emu for emulation

src_host   := host.cpp
src_kernel := kernels/conv2d_kernel.cpp

xclbin := conv2d.xclbin
host   := host

CXXFLAGS := -std=c++17 -I$(XILINX_HLS)/include -Wall
LDFLAGS  := -L$(XILINX_HLS)/lib -lOpenCL -lpthread

all: $(xclbin) $(host)

$(xclbin): $(src_kernel) data/weights_bias.h kernels/aes_utils.h
	v++ -c -t $(TARGET) --platform $(PLATFORM)  $< -o kernel.o
	v++ -l -t $(TARGET) --platform $(PLATFORM)  kernel.o --config config.ini -o $@

$(host): $(src_host)
	g++ $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: all
	./$(host) $(xclbin)

clean:
	rm -rf *.o *.xclbin _x $(host)
