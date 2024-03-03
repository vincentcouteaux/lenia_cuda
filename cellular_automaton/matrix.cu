
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix.cuh"
#include <stdio.h>

#include <time.h>
#include <stdlib.h>

//srand(time(NULL));
//int r = rand(); 



__host__ __device__ int getIdx(Matrix mat, int x, int y) {
	return y * mat.width + x;
}

float getValueHost(Matrix mat, int x, int y) {
	const int index = getIdx(mat, x, y);
	return mat.host_data[index];
}

cudaError_t copyHost2Device(Matrix* mat, bool free_ptr=false) {
	if (free_ptr) cudaFree(mat->device_data);
	cudaMalloc((void**)&mat->device_data, mat->width * mat->height * sizeof(float));
	return cudaMemcpy(mat->device_data, mat->host_data, mat->width * mat->height * sizeof(float), cudaMemcpyHostToDevice);
}
void copyDevice2Host(Matrix* mat, bool free_ptr=false) {
	//cudaMalloc((void**)&mat.device_data, mat.width * mat.height * sizeof(float));
	if (free_ptr) free(mat->host_data); //doesn't work: access violation at location 0xFFFFFFFFFFFF
	mat->host_data = (float*)malloc(mat->width * mat->height * sizeof(float));
	cudaMemcpy(mat->host_data, mat->device_data, mat->width * mat->height * sizeof(float), cudaMemcpyDeviceToHost);
}

void printMatrix(Matrix mat) {
	printf("[");
	for (int i = 0; i < mat.width; i++) {
		printf("[");
		for (int j = 0; j < mat.height; j++) {
			printf("%d", (int)getValueHost(mat, i, j));
			if (j < mat.height - 1) {
				printf(", ");
			}
		}
		printf("]");
		if (i < mat.width - 1) {
			printf(",\n");
		}
	}
	printf("]");
}

Matrix getRandomMatrix(int width, int height) {
	Matrix mat;
	mat.width = width;
	mat.height = height;
	mat.host_data = (float*)malloc(mat.width * mat.height * sizeof(float));
	for (int i = 0; i < mat.width; i++) {
		for (int j = 0; j < mat.width; j++) {
			mat.host_data[getIdx(mat, i, j)] = (float) (rand() % 20);
		}
	}
	//mat.device_data = 0;
	return mat;
}

/*
__global__ void addMatrixKer(Matrix mat1, Matrix mat2, Matrix result) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < mat1.width*mat1.height) result.device_data[index] = mat1.device_data[index] + mat2.device_data[index];
}

Matrix addMatrix(Matrix mat1, Matrix mat2) {
	Matrix result;
	result.width = mat1.width;
	result.height = mat1.height;
	cudaMalloc((void**)result.device_data, result.width * result.height * sizeof(float));
	addMatrixKer<<<16, (result.width * result.height + 1) / 16 >>> (mat1, mat2, result);
	copyDevice2Host(&result);
	return result;
}


*/
