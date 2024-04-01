#include "Header.cuh"

void test_sum() {
	int N = 40001;
	float* ones_host = (float*) malloc(N*sizeof(float));
	for (int i = 0; i < N; i++) {
		ones_host[i] = 2.0f;
	}

	float* ones;
	cudaMalloc((void**)&ones, N * sizeof(float));
	//cudaMemset(ones, 1.0f, N * sizeof(float));
	cudaMemcpy(ones, ones_host, N * sizeof(float), cudaMemcpyHostToDevice);
	float sum = launch_reduceSum(ones, N);
}
