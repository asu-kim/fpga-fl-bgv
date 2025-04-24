// Auto-generated weights & biases with easy-to-test values
#ifndef WEIGHTS_BIAS_H
#define WEIGHTS_BIAS_H

#include <cstdint>

static const int32_t CONV1_BIAS_INT8_DATA[6] = {
  10, -10, 5, -5, 50, -40
};
static const int CONV1_BIAS_INT8_SHAPE[] = { 4 };

// static const int32_t CONV1_WEIGHT_INT8_DATA[25] = {
//   -128, -100, -50, -10, 0,
//   0, 10, 20, 30, 40,
//   50, 60, 70, 80, 90,
//   100, 110, 120, 125, 126,
//   127, 127, 127, 127, 127
// };
static const int32_t CONV1_WEIGHT_INT8_DATA[150] = {
  -128, -100, -50, -10, 0,
  0, 10, 50, 100, 1,
  2, 3, 4, 5, 6,
  7, 8, 9, 10, 0,
  0, 0, 0, 0, 127,

  0, -100, -50, -10, 0,
  0, 10, 50, 100, 1,
  2, 3, 4, 5, 6,
  7, 8, 9, 10, 0,
  0, 0, 0, 0, 0,

  0, 0, 0, 0, 0,
  0, 0, 0, 0, 1,
  2, 3, 4, 5, 6,
  7, 8, 9, 10, 0,
  0, 0, 0, 0, 0, 

  0, 0, 0, 0, 0,
  0, 0, 0, 0, 1,
  2, 3, 4, 5, 6,
  7, 8, 9, 10, 0,
  0, 0, 0, 0, 0, 

  5, 0, 0, 0, 0,
  0, 0, 10, 0, 1,
  2, 3, 4, 5, 6,
  7, 8, 9, 10, 0,
  0, 0, 0, 0, 0, 

  -5, 0, 0, 0, 0,
  0, 0, 0, 0, 1,
  2, 3, 4, 5, 6,
  7, 8, 9, 10, 0,
  0, 0, 0, 0, 0
};
static const int CONV1_WEIGHT_INT8_SHAPE[] = { 4, 1, 5, 5 };

static const int32_t FC1_BIAS_INT8_DATA[5] = {
  0, 1, 2, 3, 4
};
static const int FC1_BIAS_INT8_SHAPE[] = { 5 };

static const int32_t FC1_WEIGHT_INT8_DATA[45] = {
  // First row: sequential values from -128 to -113
  -128, -127, -126, -125, -124, -123, -122, -121, -120,
  
  // Second row: sequential values from -112 to -97
  -112, -111, -110, -109, -108, -107, -106, -105, -104,
  
  // Third row: sequential values from -96 to -81
  -96, -95, -94, -93, -92, -91, -90, -89, -88,
  
  // Fourth row: sequential values from -80 to -65
  -80, -79, -78, -77, -76, -75, -74, -73, -72,
  
  // Fifth row: sequential values from -64 to -49
  -64, -63, -62, -61, -60, -59, -58, -57, -56,
};
static const int FC1_WEIGHT_INT8_SHAPE[] = { 45 };

#endif // WEIGHTS_BIAS_H