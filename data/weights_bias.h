// Auto-generated weights & biases with easy-to-test values
#ifndef WEIGHTS_BIAS_H
#define WEIGHTS_BIAS_H

#include <cstdint>

static const int32_t CONV1_BIAS_INT8_DATA[6] = {
  14,  -106,  -102,  -68,  122,  -96,
};
static const int CONV1_BIAS_INT8_SHAPE[] = { 6 };

static const int32_t CONV1_WEIGHT_INT8_DATA[150] = {
  -6,  8,  -3,  23,  24, 
  -10,  18,  7,  12,  -28,
  6,  22, 127,  -4,  -13,  
  -9,  20,  7,  -44,  -43,
  -18,  9,  -14,  -5,  -6,

  -3,  10,  -24,  -13,  -16,
  -128,  -3,  8,  -2,  -41,
  -14,  -1,  -20,  -3,  -43,
  -37,  8,  -9,  -60,  6,
  -26,  -41,  -18,  -5,  54,

  2,  -9,  -66,  -74,  -11,
  40,  50,  -4,  -128,  -55,
  46,  55,  46,  18,  0,
  8, 38,  37,  28,  16, 
  -16,  14, -4,  -16,  5, 

  2,  19,  12, 6,  6, 
  24,  22,  44, 31,  15,  
  -23,  -14,  5, 12,  16,
  -45,  -127,  -68, -40,  9,
  5,  8,  -12, -9,  -22,

  127,  12,  -19, -67,  -9,  
  48,  -18,  -64, -43,  31, 
  46,  -78,  -54, -29,  23,  
  -38,  -82,  -17, -7,  45,
  -50,  -43,  -16, 5,  21,

  27,  8,  -30, -15,  -13,
  -49,  -128,  -35, 1,  8,
  -54,  -24,  8, 38,  6, 
  24,  49,  53,  30, -19,  
  33,  45,  20,  -1,  -40,
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