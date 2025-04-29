#include "hls_signal_handler.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <vector>
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_directio.h"
#include "hls_stream.h"
using namespace std;

namespace hls::sim
{
  template<size_t n>
  struct Byte {
    unsigned char a[n];

    Byte()
    {
      for (size_t i = 0; i < n; ++i) {
        a[i] = 0;
      }
    }

    template<typename T>
    Byte<n>& operator= (const T &val)
    {
      std::memcpy(a, &val, n);
      return *this;
    }
  };

  struct SimException : public std::exception {
    const std::string msg;
    const size_t line;
    SimException(const std::string &msg, const size_t line)
      : msg(msg), line(line)
    {
    }
  };

  void errExit(const size_t line, const std::string &msg)
  {
    std::string s;
    s += "ERROR";
//  s += '(';
//  s += __FILE__;
//  s += ":";
//  s += std::to_string(line);
//  s += ')';
    s += ": ";
    s += msg;
    s += "\n";
    fputs(s.c_str(), stderr);
    exit(1);
  }
}


namespace hls::sim
{
  struct Buffer {
    char *first;
    Buffer(char *addr) : first(addr)
    {
    }
  };

  struct DBuffer : public Buffer {
    static const size_t total = 1<<10;
    size_t ufree;

    DBuffer(size_t usize) : Buffer(nullptr), ufree(total)
    {
      first = new char[usize*ufree];
    }

    ~DBuffer()
    {
      delete[] first;
    }
  };

  struct CStream {
    char *front;
    char *back;
    size_t num;
    size_t usize;
    std::list<Buffer*> bufs;
    bool dynamic;

    CStream() : front(nullptr), back(nullptr),
                num(0), usize(0), dynamic(true)
    {
    }

    ~CStream()
    {
      for (Buffer *p : bufs) {
        delete p;
      }
    }

    template<typename T>
    T* data()
    {
      return (T*)front;
    }

    template<typename T>
    void transfer(hls::stream<T> *param)
    {
      while (!empty()) {
        param->write(*(T*)nextRead());
      }
    }

    bool empty();
    char* nextRead();
    char* nextWrite();
  };

  bool CStream::empty()
  {
    return num == 0;
  }

  char* CStream::nextRead()
  {
    assert(num > 0);
    char *res = front;
    front += usize;
    if (dynamic) {
      if (++static_cast<DBuffer*>(bufs.front())->ufree == DBuffer::total) {
        if (bufs.size() > 1) {
          bufs.pop_front();
          front = bufs.front()->first;
        } else {
          front = back = bufs.front()->first;
        }
      }
    }
    --num;
    return res;
  }

  char* CStream::nextWrite()
  {
    if (dynamic) {
      if (static_cast<DBuffer*>(bufs.back())->ufree == 0) {
        bufs.push_back(new DBuffer(usize));
        back = bufs.back()->first;
      }
      --static_cast<DBuffer*>(bufs.back())->ufree;
    }
    char *res = back;
    back += usize;
    ++num;
    return res;
  }

  std::list<CStream> streams;
  std::map<char*, CStream*> prebuilt;

  CStream* createStream(size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = true;
      s.bufs.push_back(new DBuffer(usize));
      s.front = s.bufs.back()->first;
      s.back = s.front;
      s.num = 0;
      s.usize = usize;
    }
    return &s;
  }

  template<typename T>
  CStream* createStream(hls::stream<T> *param)
  {
    CStream *s = createStream(sizeof(T));
    {
      s->dynamic = true;
      while (!param->empty()) {
        T data = param->read();
        memcpy(s->nextWrite(), (char*)&data, sizeof(T));
      }
      prebuilt[s->front] = s;
    }
    return s;
  }

  template<typename T>
  CStream* createStream(T *param, size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = false;
      s.bufs.push_back(new Buffer((char*)param));
      s.front = s.back = s.bufs.back()->first;
      s.usize = usize;
      s.num = ~0UL;
    }
    prebuilt[s.front] = &s;
    return &s;
  }

  CStream* findStream(char *buf)
  {
    return prebuilt.at(buf);
  }
}
class AESL_RUNTIME_BC {
  public:
    AESL_RUNTIME_BC(const char* name) {
      file_token.open( name);
      if (!file_token.good()) {
        cout << "Failed to open tv file " << name << endl;
        exit (1);
      }
      file_token >> mName;//[[[runtime]]]
    }
    ~AESL_RUNTIME_BC() {
      file_token.close();
    }
    int read_size () {
      int size = 0;
      file_token >> mName;//[[transaction]]
      file_token >> mName;//transaction number
      file_token >> mName;//pop_size
      size = atoi(mName.c_str());
      file_token >> mName;//[[/transaction]]
      return size;
    }
  public:
    fstream file_token;
    string mName;
};
using hls::sim::Byte;
struct __cosim_s32__ { char data[32]; };
struct __cosim_s64__ { char data[64]; };
extern "C" void train_lenet5_top(Byte<32>*, Byte<32>*, Byte<64>*, Byte<64>*, Byte<64>*, Byte<64>*, Byte<64>*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
extern "C" void apatb_train_lenet5_top_hw(volatile void * __xlx_apatb_param_image_r, volatile void * __xlx_apatb_param_conv1_weight, volatile void * __xlx_apatb_param_conv1_bias, volatile void * __xlx_apatb_param_conv2_in, volatile void * __xlx_apatb_param_conv2_weight, volatile void * __xlx_apatb_param_conv2_bias, volatile void * __xlx_apatb_param_fc1_in, volatile void * __xlx_apatb_param_fc1_weight, volatile void * __xlx_apatb_param_fc1_bias, volatile void * __xlx_apatb_param_fc2_in, volatile void * __xlx_apatb_param_fc2_weight, volatile void * __xlx_apatb_param_fc2_bias, volatile void * __xlx_apatb_param_fc3_in, volatile void * __xlx_apatb_param_fc3_weight, volatile void * __xlx_apatb_param_fc3_bias, volatile void * __xlx_apatb_param_probs, volatile void * __xlx_apatb_param_label_r) {
using hls::sim::createStream;
  // Collect __xlx_image_r__tmp_vec
std::vector<Byte<32>> __xlx_image_r__tmp_vec;
for (size_t i = 0; i < 98; ++i){
__xlx_image_r__tmp_vec.push_back(((Byte<32>*)__xlx_apatb_param_image_r)[i]);
}
  int __xlx_size_param_image_r = 98;
  int __xlx_offset_param_image_r = 0;
  int __xlx_offset_byte_param_image_r = 0*32;
  // Collect __xlx_conv1_weight_conv1_bias__tmp_vec
std::vector<Byte<32>> __xlx_conv1_weight_conv1_bias__tmp_vec;
for (size_t i = 0; i < 19; ++i){
__xlx_conv1_weight_conv1_bias__tmp_vec.push_back(((Byte<32>*)__xlx_apatb_param_conv1_weight)[i]);
}
  int __xlx_size_param_conv1_weight = 19;
  int __xlx_offset_param_conv1_weight = 0;
  int __xlx_offset_byte_param_conv1_weight = 0*32;
for (size_t i = 0; i < 1; ++i){
__xlx_conv1_weight_conv1_bias__tmp_vec.push_back(((Byte<32>*)__xlx_apatb_param_conv1_bias)[i]);
}
  int __xlx_size_param_conv1_bias = 1;
  int __xlx_offset_param_conv1_bias = 19;
  int __xlx_offset_byte_param_conv1_bias = 19*32;
  // Collect __xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec
std::vector<Byte<64>> __xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec;
for (size_t i = 0; i < 216; ++i){
__xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_conv2_in)[i]);
}
  int __xlx_size_param_conv2_in = 216;
  int __xlx_offset_param_conv2_in = 0;
  int __xlx_offset_byte_param_conv2_in = 0*64;
for (size_t i = 0; i < 150; ++i){
__xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_conv2_weight)[i]);
}
  int __xlx_size_param_conv2_weight = 150;
  int __xlx_offset_param_conv2_weight = 216;
  int __xlx_offset_byte_param_conv2_weight = 216*64;
for (size_t i = 0; i < 1; ++i){
__xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_conv2_bias)[i]);
}
  int __xlx_size_param_conv2_bias = 1;
  int __xlx_offset_param_conv2_bias = 366;
  int __xlx_offset_byte_param_conv2_bias = 366*64;
  // Collect __xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec
std::vector<Byte<64>> __xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec;
for (size_t i = 0; i < 16; ++i){
__xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc1_in)[i]);
}
  int __xlx_size_param_fc1_in = 16;
  int __xlx_offset_param_fc1_in = 0;
  int __xlx_offset_byte_param_fc1_in = 0*64;
for (size_t i = 0; i < 1920; ++i){
__xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc1_weight)[i]);
}
  int __xlx_size_param_fc1_weight = 1920;
  int __xlx_offset_param_fc1_weight = 16;
  int __xlx_offset_byte_param_fc1_weight = 16*64;
for (size_t i = 0; i < 8; ++i){
__xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc1_bias)[i]);
}
  int __xlx_size_param_fc1_bias = 8;
  int __xlx_offset_param_fc1_bias = 1936;
  int __xlx_offset_byte_param_fc1_bias = 1936*64;
  // Collect __xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec
std::vector<Byte<64>> __xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec;
for (size_t i = 0; i < 8; ++i){
__xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc2_in)[i]);
}
  int __xlx_size_param_fc2_in = 8;
  int __xlx_offset_param_fc2_in = 0;
  int __xlx_offset_byte_param_fc2_in = 0*64;
for (size_t i = 0; i < 630; ++i){
__xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc2_weight)[i]);
}
  int __xlx_size_param_fc2_weight = 630;
  int __xlx_offset_param_fc2_weight = 8;
  int __xlx_offset_byte_param_fc2_weight = 8*64;
for (size_t i = 0; i < 6; ++i){
__xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc2_bias)[i]);
}
  int __xlx_size_param_fc2_bias = 6;
  int __xlx_offset_param_fc2_bias = 638;
  int __xlx_offset_byte_param_fc2_bias = 638*64;
  // Collect __xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec
std::vector<Byte<64>> __xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec;
for (size_t i = 0; i < 6; ++i){
__xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc3_in)[i]);
}
  int __xlx_size_param_fc3_in = 6;
  int __xlx_offset_param_fc3_in = 0;
  int __xlx_offset_byte_param_fc3_in = 0*64;
for (size_t i = 0; i < 53; ++i){
__xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc3_weight)[i]);
}
  int __xlx_size_param_fc3_weight = 53;
  int __xlx_offset_param_fc3_weight = 6;
  int __xlx_offset_byte_param_fc3_weight = 6*64;
for (size_t i = 0; i < 1; ++i){
__xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_fc3_bias)[i]);
}
  int __xlx_size_param_fc3_bias = 1;
  int __xlx_offset_param_fc3_bias = 59;
  int __xlx_offset_byte_param_fc3_bias = 59*64;
  // Collect __xlx_probs_label_r__tmp_vec
std::vector<Byte<64>> __xlx_probs_label_r__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_probs_label_r__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_probs)[i]);
}
  int __xlx_size_param_probs = 1;
  int __xlx_offset_param_probs = 0;
  int __xlx_offset_byte_param_probs = 0*64;
for (size_t i = 0; i < 1; ++i){
__xlx_probs_label_r__tmp_vec.push_back(((Byte<64>*)__xlx_apatb_param_label_r)[i]);
}
  int __xlx_size_param_label_r = 1;
  int __xlx_offset_param_label_r = 1;
  int __xlx_offset_byte_param_label_r = 1*64;
  // DUT call
  train_lenet5_top(__xlx_image_r__tmp_vec.data(), __xlx_conv1_weight_conv1_bias__tmp_vec.data(), __xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec.data(), __xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec.data(), __xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec.data(), __xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec.data(), __xlx_probs_label_r__tmp_vec.data(), __xlx_offset_byte_param_image_r, __xlx_offset_byte_param_conv1_weight, __xlx_offset_byte_param_conv1_bias, __xlx_offset_byte_param_conv2_in, __xlx_offset_byte_param_conv2_weight, __xlx_offset_byte_param_conv2_bias, __xlx_offset_byte_param_fc1_in, __xlx_offset_byte_param_fc1_weight, __xlx_offset_byte_param_fc1_bias, __xlx_offset_byte_param_fc2_in, __xlx_offset_byte_param_fc2_weight, __xlx_offset_byte_param_fc2_bias, __xlx_offset_byte_param_fc3_in, __xlx_offset_byte_param_fc3_weight, __xlx_offset_byte_param_fc3_bias, __xlx_offset_byte_param_probs, __xlx_offset_byte_param_label_r);
// print __xlx_apatb_param_image_r
for (size_t i = 0; i < __xlx_size_param_image_r; ++i) {
((Byte<32>*)__xlx_apatb_param_image_r)[i] = __xlx_image_r__tmp_vec[__xlx_offset_param_image_r+i];
}
// print __xlx_apatb_param_conv1_weight
for (size_t i = 0; i < __xlx_size_param_conv1_weight; ++i) {
((Byte<32>*)__xlx_apatb_param_conv1_weight)[i] = __xlx_conv1_weight_conv1_bias__tmp_vec[__xlx_offset_param_conv1_weight+i];
}
// print __xlx_apatb_param_conv1_bias
for (size_t i = 0; i < __xlx_size_param_conv1_bias; ++i) {
((Byte<32>*)__xlx_apatb_param_conv1_bias)[i] = __xlx_conv1_weight_conv1_bias__tmp_vec[__xlx_offset_param_conv1_bias+i];
}
// print __xlx_apatb_param_conv2_in
for (size_t i = 0; i < __xlx_size_param_conv2_in; ++i) {
((Byte<64>*)__xlx_apatb_param_conv2_in)[i] = __xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec[__xlx_offset_param_conv2_in+i];
}
// print __xlx_apatb_param_conv2_weight
for (size_t i = 0; i < __xlx_size_param_conv2_weight; ++i) {
((Byte<64>*)__xlx_apatb_param_conv2_weight)[i] = __xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec[__xlx_offset_param_conv2_weight+i];
}
// print __xlx_apatb_param_conv2_bias
for (size_t i = 0; i < __xlx_size_param_conv2_bias; ++i) {
((Byte<64>*)__xlx_apatb_param_conv2_bias)[i] = __xlx_conv2_in_conv2_weight_conv2_bias__tmp_vec[__xlx_offset_param_conv2_bias+i];
}
// print __xlx_apatb_param_fc1_in
for (size_t i = 0; i < __xlx_size_param_fc1_in; ++i) {
((Byte<64>*)__xlx_apatb_param_fc1_in)[i] = __xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec[__xlx_offset_param_fc1_in+i];
}
// print __xlx_apatb_param_fc1_weight
for (size_t i = 0; i < __xlx_size_param_fc1_weight; ++i) {
((Byte<64>*)__xlx_apatb_param_fc1_weight)[i] = __xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec[__xlx_offset_param_fc1_weight+i];
}
// print __xlx_apatb_param_fc1_bias
for (size_t i = 0; i < __xlx_size_param_fc1_bias; ++i) {
((Byte<64>*)__xlx_apatb_param_fc1_bias)[i] = __xlx_fc1_in_fc1_weight_fc1_bias__tmp_vec[__xlx_offset_param_fc1_bias+i];
}
// print __xlx_apatb_param_fc2_in
for (size_t i = 0; i < __xlx_size_param_fc2_in; ++i) {
((Byte<64>*)__xlx_apatb_param_fc2_in)[i] = __xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec[__xlx_offset_param_fc2_in+i];
}
// print __xlx_apatb_param_fc2_weight
for (size_t i = 0; i < __xlx_size_param_fc2_weight; ++i) {
((Byte<64>*)__xlx_apatb_param_fc2_weight)[i] = __xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec[__xlx_offset_param_fc2_weight+i];
}
// print __xlx_apatb_param_fc2_bias
for (size_t i = 0; i < __xlx_size_param_fc2_bias; ++i) {
((Byte<64>*)__xlx_apatb_param_fc2_bias)[i] = __xlx_fc2_in_fc2_weight_fc2_bias__tmp_vec[__xlx_offset_param_fc2_bias+i];
}
// print __xlx_apatb_param_fc3_in
for (size_t i = 0; i < __xlx_size_param_fc3_in; ++i) {
((Byte<64>*)__xlx_apatb_param_fc3_in)[i] = __xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec[__xlx_offset_param_fc3_in+i];
}
// print __xlx_apatb_param_fc3_weight
for (size_t i = 0; i < __xlx_size_param_fc3_weight; ++i) {
((Byte<64>*)__xlx_apatb_param_fc3_weight)[i] = __xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec[__xlx_offset_param_fc3_weight+i];
}
// print __xlx_apatb_param_fc3_bias
for (size_t i = 0; i < __xlx_size_param_fc3_bias; ++i) {
((Byte<64>*)__xlx_apatb_param_fc3_bias)[i] = __xlx_fc3_in_fc3_weight_fc3_bias__tmp_vec[__xlx_offset_param_fc3_bias+i];
}
// print __xlx_apatb_param_probs
for (size_t i = 0; i < __xlx_size_param_probs; ++i) {
((Byte<64>*)__xlx_apatb_param_probs)[i] = __xlx_probs_label_r__tmp_vec[__xlx_offset_param_probs+i];
}
// print __xlx_apatb_param_label_r
for (size_t i = 0; i < __xlx_size_param_label_r; ++i) {
((Byte<64>*)__xlx_apatb_param_label_r)[i] = __xlx_probs_label_r__tmp_vec[__xlx_offset_param_label_r+i];
}
}
