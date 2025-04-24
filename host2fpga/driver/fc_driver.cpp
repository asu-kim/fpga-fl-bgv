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
    cl::Kernel kernel(program, "fc_kernel");

    // Generate test data & encrypt
    extern uint32_t FC1_WEIGHT_INT8_SHAPE[];
    constexpr int IN_DIM=FC1_WEIGHT_INT8_SHAPE[1], OUT_DIM=FC1_WEIGHT_INT8_SHAPE[0];

    std::vector<uint32_t> input(IN_DIM);
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(-128,127);
    for(auto &v:input) v = encrypt((uint32_t)dist(rng));

    // flatten weights from weights_bias.h
    extern uint32_t FC1_WEIGHT_INT8_DATA[]; // provided by header
    extern uint32_t FC1_BIAS_INT8_DATA[];
    const int W_TOTAL = IN_DIM * OUT_DIM;
    std::vector<uint32_t> enc_w(W_TOTAL);
    std::vector<uint32_t> enc_b(OUT_DIM);
    for(int i=0;i<W_TOTAL;i++) enc_w[i]=encrypt(FC1_WEIGHT_INT8_DATA[i]);
    for(int i=0;i<OUT_DIM;i++)  enc_b[i]=encrypt(FC1_BIAS_INT8_DATA[i]);

    std::vector<uint32_t> enc_out(OUT_DIM);

    // Allocate device buffers (HBM banks via EXT_PTR)
    cl_int err;
    cl::Buffer buf_w(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(uint32_t)*enc_w.size(), enc_w.data(), &err);
    cl::Buffer buf_b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(uint32_t)*enc_b.size(), enc_b.data(), &err);
    cl::Buffer buf_in(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(uint32_t)*input.size(), input.data(), &err);
    cl::Buffer buf_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(uint32_t)*enc_out.size(), enc_out.data(), &err);
    bool use_relu = true;

    // Set kernel args & run
    kernel.setArg(0, buf_w);
    kernel.setArg(1, buf_b);
    kernel.setArg(2, buf_in);
    kernel.setArg(3, buf_out);
    kernel.setArg(4, use_relu);

    q.enqueueMigrateMemObjects({buf_w,buf_b,buf_in,use_relu}, 0/*host→device*/);
    q.enqueueTask(kernel);
    q.enqueueMigrateMemObjects({buf_out}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    // Decrypt and check one sample value
    auto first = decrypt(enc_out[0]);
    std::cout << "First output (decrypted) = " << first << std::endl;
    return 0;
}

