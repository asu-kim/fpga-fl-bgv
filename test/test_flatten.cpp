#include <iostream>
#include <cmath>
#include "flatten.h"
#include "test_utils.h"

#ifndef __SYNTHESIS__
int main() {
    hls::stream<data_t> in_stream, out_stream;

    int errs = 0;
    // test 1: 2d to 1d
    {
        const std::array<data_t, 2*3> input_data = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };

        for(const auto& val : input_data) {
            in_stream.write(val);
        }

        flatten<2,3>(in_stream, out_stream);

        for(int i=0; i<input_data.size(); ++i) {
            data_t expected = input_data[i];
            data_t actual = out_stream.read();

            if(actual != expected) {
                std::cerr << "2D Test: Mismatch at index " << i
                    << " - Expected: " << expected.to_double()
                    << " , Actual: " << actual.to_double() << std::endl;
            }
        }
    }

    // test 2: 3d to 1d
    {
        const std::array<data_t, 2*2*2> input_data = {
            1.0, 2.0, 3.0,
            4.0, 5.4, 6.0,
            7.0, 8.0
        };

        for(const auto& val : input_data) {
            in_stream.write(val);
        }

        flatten<2,2,2>(in_stream, out_stream);

        for(int i=0; i<input_data.size(); ++i) {
            data_t expected = input_data[i];
            data_t actual = out_stream.read();

            if(actual != expected) {
                std::cerr << "3D Test: Mismatch at index " << i
                    << " - Expected: " << expected.to_double()
                    << " , Actual: " << actual.to_double() << std::endl;
            }
        }
    }


    std::cout << "Flatten Test: " << (errs ? "Fail" : "Pass") 
        << "(" << errs << " errors)\n";

    return errs;
}

#endif
