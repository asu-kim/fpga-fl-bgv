#include <CL/cl2.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>

#include "weights_bias.h"   

static uint32_t XOR_KEY = 0x55AA55AAu;
inline uint32_t encrypt(uint32_t x){ return x ^ XOR_KEY; }
inline uint32_t decrypt(uint32_t x){ return x ^ XOR_KEY; }

int main(int argc, char** argv) {
    if(argc < 2){ std::cerr << "Usage: host <xclbin>" << std::endl; return 1; }
    std::string binaryFile = argv[1];

    // OpenCL boilerplate: pick first Xilinx platform/device
    auto platforms = cl::Platform::get();
    cl::Platform platform;
    for(auto &p: platforms){ std::string name; p.getInfo(CL_PLATFORM_NAME,&name); if(name.find("Xilinx")!=std::string::npos){platform=p;break;} }
    auto devices = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load binary & create kernel
    std::ifstream bin(binaryFile, std::ios::binary|std::ios::in);
    std::vector<unsigned char> buf(std::istreambuf_iterator<char>(bin), {});
    cl::Program::Binaries bins{{buf.data(), buf.size()}};
    cl::Program program(context, {device}, bins);
    cl::Kernel kernel(program, "avg_pool_kernel");

    const int IN_HEIGHT=24, IN_WIDTH=24, IN_C=6;
    const int POOLSIZE = 2, STRIDE=2;

    int act_total = IN_HEIGHT*IN_WIDTH*IN_C;
    std::vector<uint32_t> input(act_total);
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(-128,127);
    for(auto &v:input) v = encrypt((uint32_t)dist(rng));

    int out_pix = (IN_HEIGHT/POOLSIZE) * (IN_WIDTH/POOLSIZE) * IN_C;
    std::vector<uint32_t> enc_out(out_pix);

    // Allocate device buffers (HBM banks via EXT_PTR)
    cl_int err;
    cl::Buffer buf_in(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(uint32_t)*input.size(), input.data(), &err);
    cl::Buffer buf_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(uint32_t)*enc_out.size(), enc_out.data(), &err);

    // Set kernel args & run
    kernel.setArg(0, buf_in);
    kernel.setArg(1, buf_out);

    q.enqueueMigrateMemObjects({buf_in}, 0/*host→device*/);
    q.enqueueTask(kernel);
    q.enqueueMigrateMemObjects({buf_out}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    // Decrypt and check one sample value
    auto first = decrypt(enc_out[0]);
    std::cout << "First output (decrypted) = " << first << std::endl;
    return 0;
}

