//#include <helper_cuda.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include "matrix.cuh"
#include "Header.cuh"
#include <stdio.h>

#define BLOCK_SIDE 16
#define THRD_PER_BLOCK (BLOCK_SIDE*BLOCK_SIDE)
#define GRID_SIZE(x) ((x + THRD_PER_BLOCK - 1)/THRD_PER_BLOCK)

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}

__device__ float computeGaussian(int imgw, float center_x, float center_y, float sigma2) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    float distx = (center_x - x);
    distx *= distx;
    float disty = (center_y - y);
    disty *= disty;
    return __expf(-(distx + disty) / sigma2); // *255;
}

__global__ void gaussianFloat(float* out, int imgw, float center_x, float center_y, float sigma2) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    out[y * imgw + x] = computeGaussian(imgw, center_x, center_y, sigma2);
}
__global__ void squareFloat(float* out, int imgw, int x0, int y0, int x1, int y1, float value_inside, float value_outside) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        out[y * imgw + x] = value_inside;
    }
    else {
        out[y * imgw + x] = value_outside;
    }
}

__global__ void gaussianBlob(unsigned int* g_odata, int imgw, float center_x, float center_y, float sigma2) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    float value = computeGaussian(imgw, center_x, center_y, sigma2);
    g_odata[y * imgw + x] = rgbToInt(value, value, value);
}

__global__ void toRGB(unsigned int* g_odata, Matrix m) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    float value = m.device_data[y * m.width + x]*255.0f;
    //if (x == 256 && y == 256) printf("TORGB %f\n", value / 255.0f);
    g_odata[y * m.width + x] = rgbToInt(value, value, value);
}

__global__ void ringKernel(Matrix m, float center_x, float center_y, float radius, float sigma2) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    float distx = (center_x - x);
    distx *= distx;
    float disty = (center_y - y);
    disty *= disty;
    float dist = sqrt(distx + disty) - radius;
    dist *= dist;
    m.device_data[y * m.width + x] = __expf(-dist / sigma2);
}

extern "C" void getRing(Matrix m, float center_x, float center_y, float radius, float sigma) {
    dim3 block(BLOCK_SIDE, BLOCK_SIDE, 1);
    dim3 grid(m.width/ block.x, m.height/ block.y, 1);
    ringKernel<<< grid, block >>> (m, center_x, center_y, radius, sigma*sigma);

}

__global__ void computeRings(Matrix m, float center_x, float center_y, int n_rings, float* radiuses, float* sigma2s, float* coefs) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    float distx = (center_x - x);
    distx *= distx;
    float disty = (center_y - y);
    disty *= disty;
    const float dist = sqrt(distx + disty);
    float out = 0;
    for (int i = 0; i < n_rings; i++) {
        const float dist2ring = dist - radiuses[i];
        out += coefs[i] * __expf(-(dist2ring*dist2ring) / sigma2s[i]);
    }
    m.device_data[y * m.width + x] = out;
}

extern "C" void launchRings(Matrix m, float center_x, float center_y, KernelParams ringParams) {
    float* sigma2s_h = (float*) malloc(ringParams.n_rings*sizeof(float));
    for (int i = 0; i < ringParams.n_rings; i++) {
        sigma2s_h[i] = ringParams.ring_sigmas[i]*ringParams.ring_sigmas[i];
    }
    float* radiuses_d;
    float* sigma2s_d;
    float* coefs_d;
    cudaMalloc((void**)&radiuses_d, ringParams.n_rings * sizeof(float));
    cudaMemcpy(radiuses_d, ringParams.ring_radiuses, ringParams.n_rings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&sigma2s_d, ringParams.n_rings * sizeof(float));
    cudaMemcpy(sigma2s_d, sigma2s_h, ringParams.n_rings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&coefs_d, ringParams.n_rings * sizeof(float));
    cudaMemcpy(coefs_d, ringParams.ring_coefs, ringParams.n_rings * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIDE, BLOCK_SIDE, 1);
    dim3 grid(m.width/ block.x, m.height/ block.y, 1);
    computeRings <<< grid, block >>> (m, center_x, center_y, ringParams.n_rings, radiuses_d, sigma2s_d, coefs_d);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "computeRings launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching computeRings!\n", cudaStatus);
    }
    cudaFree(radiuses_d);
    cudaFree(sigma2s_d);
    cudaFree(coefs_d);
    free(sigma2s_h);
}

extern "C" void launch_cudaProcess(unsigned int* g_odata, int imgw, int imgh, float sigma) {
    dim3 block(BLOCK_SIDE, BLOCK_SIDE, 1);
    dim3 grid(imgw / block.x, imgh / block.y, 1);
    //cudaProcess <<<grid, block, sbytes>>> (g_odata, imgw, frame_num);
    gaussianBlob<<<grid, block>>> (g_odata, imgw, 256, 256, sigma*sigma);
}
extern "C" void getGaussianBlob(Matrix m, float center_x, float center_y, float sigma) {
    dim3 block(BLOCK_SIDE, BLOCK_SIDE, 1);
    dim3 grid(m.width / block.x, m.height / block.y, 1);
    gaussianFloat <<< grid, block >>> (m.device_data, m.width, center_x, center_y, sigma*sigma);
}
extern "C" void getSquare(Matrix m, int x0, int y0, int x1, int y1, float value_inside, float value_outside) {
    dim3 block(BLOCK_SIDE, BLOCK_SIDE, 1);
    dim3 grid(m.width / block.x, m.height / block.y, 1);
    squareFloat<<< grid, block >>> (m.device_data, m.width, x0, y0, x1, y1, value_inside, value_outside);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
}

extern "C" void launch_toRGB(unsigned int* g_odata, Matrix m) {
    dim3 block(BLOCK_SIDE, BLOCK_SIDE, 1);
    dim3 grid(m.width / block.x, m.height / block.y, 1);
    toRGB <<<grid, block>>> (g_odata, m);
}

__global__ void multiplyComplex(float2* result, float2* vec1, float2* vec2, int size, float scale) {
    //printf("%d index\n", threadIdx.x);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index].x = (vec1[index].x * vec2[index].x - vec1[index].y * vec2[index].y)/scale;
        result[index].y = (vec1[index].y * vec2[index].x + vec1[index].x * vec2[index].y)/scale;
    }
}
extern "C" void launch_multiplyComplex(float2 * result, float2 * vec1, float2 * vec2, int size, float scale) {
    //multiplyComplex<<<256, size/256>>>(result, vec1, vec2, size, scale);
    //multiplyComplex<<<(size + THRD_PER_BLOCK - 1)/THRD_PER_BLOCK, THRD_PER_BLOCK>>>(result, vec1, vec2, size, scale);
    multiplyComplex<<<GRID_SIZE(size), THRD_PER_BLOCK>>>(result, vec1, vec2, size, scale);
    //printf("BBB %d, %d", GRID_SIZE(size), THRD_PER_BLOCK);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "AAA multiplyComplex launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching multiplyComplex!\n", cudaStatus);
    }
}

__global__ void updateState(float* stateData, float* resultConv, int size, float peakValue, float std2, float step) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    float gapFromPeak = resultConv[index] - peakValue;
    gapFromPeak *= gapFromPeak;
    if (index < size) {
        stateData[index] += (2 * __expf(-gapFromPeak / std2) - 1) * step;
        stateData[index] = clamp(stateData[index], 0.0f, 1.0f);
        //if (index == 256 * 256) {
        //    printf("UPDATE %f, %f, %f\n", stateData[index], (2 * __expf(gapFromPeak / std2) - 1), elapsedTime * step);
        //}
    }
}
extern "C" void launch_updateState(Matrix state, float* resultConv, float peakValue, float std, float elapsedTime, float step) {
    int size = state.width * state.height;
    //updateState<<<256, size / 256>>>(state.device_data, resultConv, size, peakValue, std*std, elapsedTime * step);
    updateState<<<GRID_SIZE(size), THRD_PER_BLOCK>>>(state.device_data, resultConv, size, peakValue, std * std, elapsedTime * step);
}

__global__ void fftShift2d(float* result, Matrix toShift) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * toShift.width + x;
    if (y < toShift.height / 2) {
        int index_;
        if (x < toShift.width / 2) {
            index_ = (y + toShift.height/2)*toShift.width + x + toShift.width / 2;
        }
        else {
            index_ = (y + toShift.height/2)*toShift.width + x - toShift.width / 2;
        }
		result[index_] = toShift.device_data[index];
		result[index] = toShift.device_data[index_];
    }
}
extern "C" void launch_fftShift2d(float* result, Matrix toShift) {
    dim3 block(BLOCK_SIDE, BLOCK_SIDE, 1);
    dim3 grid(toShift.width / block.x, toShift.height / block.y / 2, 1);
    fftShift2d<<<grid, block>>>(result, toShift);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fftShift launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching fftShift!\n", cudaStatus);
    }
}

__global__ void reduceSum(float* output, float* values, int size) {
    const int block_index = blockDim.x * blockIdx.x; // +threadIdx.x;
    extern __shared__ float smem[];
    smem[threadIdx.x] = values[block_index + threadIdx.x];
    __syncthreads();
    int offset = 2 * threadIdx.x;
    for (int gap = 1; gap < blockDim.x; gap <<= 1) {
        if (offset < blockDim.x) {
            smem[offset] += smem[offset + gap];
            offset <<= 1;
			__syncthreads();
        }
    }
    output[blockIdx.x] = smem[0];
}

float sumCpu(float* values, int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += values[i];
    }
    return result;
}

extern "C" float launch_reduceSum(float* values, int size) {
    const int threads_per_block = 256;
    float* toReduce = values;
    int curSize = size;
    int num_blocks = (size + threads_per_block - 1)/threads_per_block;

	float* blockSums;
	cudaMalloc((void**)&blockSums, num_blocks*sizeof(float));

    while (curSize > 1) {
        num_blocks = (curSize + threads_per_block - 1) / threads_per_block;
		reduceSum<<<num_blocks, threads_per_block, threads_per_block*sizeof(float) >>>(blockSums, toReduce, curSize);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduceSum launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching reduceSum!\n", cudaStatus);
		}
		//printf("NUM BLOCKS %d, THRD_PER_BLOCK %d\n", num_blocks, threads_per_block);
        toReduce = blockSums;
        curSize = num_blocks;
    }
    float output;
    cudaMemcpy(&output, blockSums, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(blockSums);
    return output;
}
