#ifndef __HEADER__
#define __HEADER__



#include "cuda_runtime.h"
#include "matrix.cuh"

int main_();
extern "C" void launch_cudaProcess(unsigned int* g_odata, int imgw, int imgh, float sigma);
extern "C" void getGaussianBlob(Matrix m, float center_x, float center_y, float sigma);
extern "C" void getSquare(Matrix m, int x0, int y0, int x1, int y1, float value_inside, float value_outside);
extern "C" void launch_toRGB(unsigned int* g_odata, Matrix m);
extern "C" void getRing(Matrix m, float center_x, float center_y, float radius, float sigma);
extern "C" void launch_multiplyComplex(float2 * result, float2 * vec1, float2 * vec2, int size, float scale);
extern "C" void launch_updateState(Matrix state, float* resultConv, float peakValue, float std2, float elapsedTime, float step);
extern "C" void launch_fftShift2d(float* result, Matrix toShift);
extern "C" float launch_reduceSum(float* values, int size);

#endif // !__HEADER__
