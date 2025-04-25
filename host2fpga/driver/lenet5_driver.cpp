// driver/lenet_driver.cpp
#include <CL/cl2.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>

#include "weights_bias.h"   // defines all *_INT8_SHAPE and *_INT8_DATA arrays

static uint32_t XOR_KEY = 0x55AA55AAu;
inline uint32_t encrypt(uint32_t x){ return x ^ XOR_KEY; }
inline uint32_t decrypt(uint32_t x){ return x ^ XOR_KEY; }

extern uint32_t CONV1_WEIGHT_INT8_SHAPE[];
extern uint32_t CONV1_WEIGHT_INT8_DATA[];
extern uint32_t CONV1_BIAS_INT8_DATA[];

extern uint32_t CONV2_WEIGHT_INT8_SHAPE[];
extern uint32_t CONV2_WEIGHT_INT8_DATA[];
extern uint32_t CONV2_BIAS_INT8_DATA[];

extern uint32_t FC1_WEIGHT_INT8_SHAPE[];
extern uint32_t FC1_WEIGHT_INT8_DATA[];
extern uint32_t FC1_BIAS_INT8_DATA[];

extern uint32_t FC2_WEIGHT_INT8_SHAPE[];
extern uint32_t FC2_WEIGHT_INT8_DATA[];
extern uint32_t FC2_BIAS_INT8_DATA[];

extern uint32_t FC3_WEIGHT_INT8_SHAPE[];
extern uint32_t FC3_WEIGHT_INT8_DATA[];
extern uint32_t FC3_BIAS_INT8_DATA[];

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: host <xclbin>" << std::endl;
        return 1;
    }
    std::string binaryFile = argv[1];

    // 1) OpenCL init
    auto platforms = cl::Platform::get();
    cl::Platform platform;
    for (auto &p : platforms) {
        std::string name;
        p.getInfo(CL_PLATFORM_NAME, &name);
        if (name.find("Xilinx") != std::string::npos) {
            platform = p;
            break;
        }
    }
    auto devices = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // 2) Load & compile xclbin
    std::ifstream bin(binaryFile, std::ios::binary|std::ios::in);
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(bin)),
                                    std::istreambuf_iterator<char>());
    cl::Program::Binaries bins{{buf.data(), buf.size()}};
    cl::Program program(context, {device}, bins);
    cl::Kernel kernel(program, "lenet5_top");

    // 3) Pull shape info
    constexpr int IMG_DIM     = 28;

    const int C1_OC = CONV1_WEIGHT_INT8_SHAPE[0],
              C1_IC = CONV1_WEIGHT_INT8_SHAPE[1],
              C1_K  = CONV1_WEIGHT_INT8_SHAPE[2];

    const int C2_OC = CONV2_WEIGHT_INT8_SHAPE[0],
              C2_IC = CONV2_WEIGHT_INT8_SHAPE[1],
              C2_K  = CONV2_WEIGHT_INT8_SHAPE[2];

    const int FC1_IN  = FC1_WEIGHT_INT8_SHAPE[0],
              FC1_OUT = FC1_WEIGHT_INT8_SHAPE[1];

    const int FC2_IN  = FC2_WEIGHT_INT8_SHAPE[0],
              FC2_OUT = FC2_WEIGHT_INT8_SHAPE[1];

    const int FC3_IN  = FC3_WEIGHT_INT8_SHAPE[0],
              FC3_OUT = FC3_WEIGHT_INT8_SHAPE[1];

    // 4) Prepare & encrypt host-side buffers
    // — image —
    std::vector<uint32_t> image(IMG_DIM*IMG_DIM);
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(-128,127);
    for (auto &v : image) v = encrypt((uint32_t)dist(rng));

    // — conv1 weights & bias —
    const int W1 = C1_OC*C1_IC*C1_K*C1_K;
    std::vector<uint32_t> conv1_w(W1), conv1_b(C1_OC);
    for (int i = 0; i < W1; ++i) conv1_w[i] = encrypt(CONV1_WEIGHT_INT8_DATA[i]);
    for (int i = 0; i < C1_OC; ++i) conv1_b[i] = encrypt(CONV1_BIAS_INT8_DATA[i]);

    // — conv2 weights & bias —
    const int W2 = C2_OC*C2_IC*C2_K*C2_K;
    std::vector<uint32_t> conv2_w(W2), conv2_b(C2_OC);
    for (int i = 0; i < W2; ++i) conv2_w[i] = encrypt(CONV2_WEIGHT_INT8_DATA[i]);
    for (int i = 0; i < C2_OC; ++i) conv2_b[i] = encrypt(CONV2_BIAS_INT8_DATA[i]);

    // — fc1 weights & bias —
    const int Wf1 = FC1_IN*FC1_OUT;
    std::vector<uint32_t> fc1_w(Wf1), fc1_b(FC1_OUT);
    for (int i = 0; i < Wf1; ++i) fc1_w[i] = encrypt(FC1_WEIGHT_INT8_DATA[i]);
    for (int i = 0; i < FC1_OUT; ++i) fc1_b[i] = encrypt(FC1_BIAS_INT8_DATA[i]);

    // — fc2 weights & bias —
    const int Wf2 = FC2_IN*FC2_OUT;
    std::vector<uint32_t> fc2_w(Wf2), fc2_b(FC2_OUT);
    for (int i = 0; i < Wf2; ++i) fc2_w[i] = encrypt(FC2_WEIGHT_INT8_DATA[i]);
    for (int i = 0; i < FC2_OUT; ++i) fc2_b[i] = encrypt(FC2_BIAS_INT8_DATA[i]);

    // — fc3 weights & bias —
    const int Wf3 = FC3_IN*FC3_OUT;
    std::vector<uint32_t> fc3_w(Wf3), fc3_b(FC3_OUT);
    for (int i = 0; i < Wf3; ++i) fc3_w[i] = encrypt(FC3_WEIGHT_INT8_DATA[i]);
    for (int i = 0; i < FC3_OUT; ++i) fc3_b[i] = encrypt(FC3_BIAS_INT8_DATA[i]);

    // — output buffer —
    std::vector<uint32_t> enc_logits(FC3_OUT);

    // 5) Allocate OpenCL buffers
    cl_int err;
    cl::Buffer buf_img   (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*image.size(),    image.data(),    &err);
    cl::Buffer buf_c1_w  (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*conv1_w.size(),   conv1_w.data(),  &err);
    cl::Buffer buf_c1_b  (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*conv1_b.size(),   conv1_b.data(),  &err);
    cl::Buffer buf_c2_w  (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*conv2_w.size(),   conv2_w.data(),  &err);
    cl::Buffer buf_c2_b  (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*conv2_b.size(),   conv2_b.data(),  &err);
    cl::Buffer buf_fc1_w (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*fc1_w.size(),     fc1_w.data(),    &err);
    cl::Buffer buf_fc1_b (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*fc1_b.size(),     fc1_b.data(),    &err);
    cl::Buffer buf_fc2_w (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*fc2_w.size(),     fc2_w.data(),    &err);
    cl::Buffer buf_fc2_b (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*fc2_b.size(),     fc2_b.data(),    &err);
    cl::Buffer buf_fc3_w (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*fc3_w.size(),     fc3_w.data(),    &err);
    cl::Buffer buf_fc3_b (context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                          sizeof(uint32_t)*fc3_b.size(),     fc3_b.data(),    &err);
    cl::Buffer buf_logit (context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY,
                          sizeof(uint32_t)*enc_logits.size(),enc_logits.data(),&err);

    // 6) Set kernel args (in the same order as top-level signature)
    kernel.setArg( 0, buf_img);
    kernel.setArg( 1, buf_c1_w);
    kernel.setArg( 2, buf_c1_b);
    kernel.setArg( 3, buf_c2_w);
    kernel.setArg( 4, buf_c2_b);
    kernel.setArg( 5, buf_fc1_w);
    kernel.setArg( 6, buf_fc1_b);
    kernel.setArg( 7, buf_fc2_w);
    kernel.setArg( 8, buf_fc2_b);
    kernel.setArg( 9, buf_fc3_w);
    kernel.setArg(10, buf_fc3_b);
    kernel.setArg(11, buf_logit);

    // 7) Migrate inputs → device, run, migrate outputs → host
    q.enqueueMigrateMemObjects({
        buf_img,
        buf_c1_w, buf_c1_b,
        buf_c2_w, buf_c2_b,
        buf_fc1_w,buf_fc1_b,
        buf_fc2_w,buf_fc2_b,
        buf_fc3_w,buf_fc3_b
    }, 0 /* H→D */);

    q.enqueueTask(kernel);

    q.enqueueMigrateMemObjects({ buf_logit },
                               CL_MIGRATE_MEM_OBJECT_HOST /* D→H */);
    q.finish();

    // 8) Decrypt & print logits
    std::cout << "Logits: ";
    for (int i = 0; i < FC3_OUT; ++i) {
        uint32_t v = decrypt(enc_logits[i]);
        std::cout << v << (i+1<FC3_OUT ? ", " : "\n");
    }

    return 0;
}
