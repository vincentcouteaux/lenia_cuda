#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "matrix.cuh"
//#include "gl_utils.h"
//#include "GL/wglew.h"
//#include "GL/freeglut.h"

__global__ void addMatrixKer(Matrix mat1, Matrix mat2, Matrix result) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < mat1.width * mat1.height) result.device_data[index] = mat1.device_data[index] + mat2.device_data[index];
}

Matrix addMatrix(Matrix mat1, Matrix mat2) {
	Matrix result;
	result.width = mat1.width;
	result.height = mat1.height;
	result.host_data = NULL;
	cudaMalloc((void**)&result.device_data, result.width * result.height * sizeof(float));
	cudaError_t err1 = copyHost2Device(&mat1, false);
	cudaError_t err2 = copyHost2Device(&mat2, false);

	addMatrixKer <<<16, result.width * result.height/16 + 1>>> (mat1, mat2, result);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaDeviceSynchronize returned error code %d after launching addMatrixKer!\n\n", cudaStatus);
	}
	copyDevice2Host(&result, false);
	return result;
}

__global__ void equalsKer(Matrix mat1, Matrix mat2, int* result) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < mat1.width * mat1.height &&
		mat1.device_data[index] != mat2.device_data[index]) {
		atomicCAS(result, 1, 0); //might be very slow... do reduce instead ?
	}
}
int equals(Matrix mat1, Matrix mat2) {
	int* device_result;
	int host_result = 1;
	cudaMalloc((void**)&device_result, sizeof(float));
	cudaMemcpy(device_result, &host_result, sizeof(int), cudaMemcpyHostToDevice);

	cudaError_t err1 = copyHost2Device(&mat1, false);
	cudaError_t err2 = copyHost2Device(&mat2, false);
	equalsKer <<<16, mat1.width * mat1.height/16 + 1>>> (mat1, mat2, device_result);
	cudaMemcpy(&host_result, device_result, sizeof(int), cudaMemcpyDeviceToHost);
	return host_result;
}

__global__ void conv2DKer(Matrix image, Matrix convker, Matrix result) {
	int index_x = threadIdx.x + blockDim.x * blockIdx.x;
	int index_y = threadIdx.y + blockDim.y * blockIdx.y;

	int thx = threadIdx.x; //Required to cast from uint
	int thy = threadIdx.y; //otherwise comparison with negative
	int bdx = blockDim.x; //fail
	int bdy = blockDim.y;

	extern __shared__ float smem[];
	float* skernel = smem + bdx * bdy;

	int smem_idx = thy * bdx + thx;
	int kernel_idx = thy * convker.width + thx;
	int image_idx = index_y * image.width + index_x;


	smem[smem_idx] = image.device_data[image_idx];
	skernel[kernel_idx] = convker.device_data[kernel_idx];

	const int halfwidth = convker.width / 2;
	const int halfheight = convker.height / 2;

	__syncthreads();
	float cur_im_value = 0;
	float cur_ker_value = 0;
	result.device_data[image_idx] = 0;
	for (int i = 0; i < convker.width; i++) {
		for (int j = 0; j < convker.height; j++) {
			if (thx >= halfwidth - i &&
				thy >= halfheight - j &&
				thx - halfwidth + i <= bdx &&
				thy - halfheight + j <= bdy) {
				cur_im_value = smem[(thy - halfheight + j) * bdx + thx - halfwidth + i];
			}
			//Use global memory when outside of block
			else if (index_x >= halfwidth - i &&
					 index_y >= halfheight - j &&
					 index_x - halfwidth + i <= image.width &&
					 index_y - halfheight + j <= image.height) {
				cur_im_value = image.device_data[(index_y - halfheight + j) * image.width + index_x - halfwidth + i];
			}
			else {
				cur_im_value = 0;
			}
			cur_ker_value = convker.device_data[j*convker.width + i];
			result.device_data[image_idx] += cur_im_value * cur_ker_value;
		}
	}
}

Matrix conv2D(Matrix image, Matrix convker) {
	Matrix result;
	result.width = image.width;
	result.height = image.height;
	cudaMalloc((void**)&result.device_data, 10*10 * sizeof(float));
	cudaError_t err1 = copyHost2Device(&image, false);
	cudaError_t err2 = copyHost2Device(&convker, false);

	dim3 gridDim;
	gridDim.x = 2;
	gridDim.y = 2;
	dim3 threadDim;
	//threadDim.x = gridDim.x / result.width;
	//threadDim.y = gridDim.y / result.height;
	threadDim.x = result.width / gridDim.x;
	threadDim.y = result.height / gridDim.y;
	const int smemSize = threadDim.x * threadDim.y + convker.width * convker.height;
	conv2DKer<<<gridDim, threadDim, smemSize>>>(image, convker, result);

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaDeviceSynchronize returned error code %d after launching conv2DKer!\n\n", cudaStatus);
	}
	copyDevice2Host(&result, false);
	return result;
}

int main__() {

	Matrix mat1 = getRandomMatrix(10,10);
	printMatrix(mat1);
	Matrix mat2 = getRandomMatrix(10,10);
	printf("\n\n");
	printMatrix(mat2);
	
	printf("\n\n");
	Matrix result = addMatrix(mat1, mat2);
	printMatrix(result);

	printf("\n\n");
	cudaFree(mat1.device_data);
	cudaFree(mat2.device_data);
	/*

	int are_equals = equals(mat1, mat2);
	if (are_equals) printf("THEY ARE EQUALS %d\n", are_equals);
	else printf("THEY ARE NOT EQUALS %d\n", are_equals);

	cudaFree(mat1.device_data);
	cudaFree(mat2.device_data);

	
	Matrix mat3;
	mat3.width = mat1.width;
	mat3.height = mat1.height;
	mat3.host_data = (float*)malloc(mat3.width * mat3.height * sizeof(float));
	cudaMemcpy(mat3.host_data, mat1.host_data, mat1.width * mat1.height * sizeof(float), cudaMemcpyHostToHost);

	are_equals = equals(mat1, mat3);
	if (are_equals) printf("THEY ARE EQUALS %d\n", are_equals);
	else printf("THEY ARE NOT EQUALS %d\n", are_equals);

	cudaFree(mat1.device_data);
	//cudaFree(mat3.device_data);
	*/

	float ker_data[] = { 1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f };
	Matrix kernel;
	kernel.width = 3;
	kernel.height = 3;
	kernel.host_data = ker_data;

	printMatrix(kernel);
	printf("\n\n");

	Matrix result_conv = conv2D(mat1, kernel);
	printf("\n");
	printMatrix(result_conv);

	free(mat1.host_data);
	free(mat2.host_data);
	free(result.host_data);
	free(result_conv.host_data);
	//free(kernel.host_data);

	
	cudaFree(result_conv.device_data);
	cudaFree(mat1.device_data);
	cudaFree(mat2.device_data);
	cudaFree(result.device_data);
	

	return 0;
}

