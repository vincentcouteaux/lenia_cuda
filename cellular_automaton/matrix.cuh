#ifndef __MATRIX__
#define __MATRIX__

struct Matrix {
	int width;
	int height;
	float* host_data;
	float* device_data;
};
int getIdx(Matrix mat, int x, int y);
float getValueHost(Matrix mat, int x, int y);
void copyDevice2Host(Matrix* mat, bool free_ptr);
cudaError_t copyHost2Device(Matrix* mat, bool free_ptr);
void printMatrix(Matrix mat);
Matrix getRandomMatrix(int width, int height);
//Matrix addMatrix(Matrix mat1, Matrix mat2);

#endif
