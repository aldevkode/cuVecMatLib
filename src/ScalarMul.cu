#include "ScalarMul.cuh"

__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float *result) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numElements) {
        sdata[tid] = vector1[index] * vector2[index];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

__global__
void MultiplyVectorsKernel(int numElements, float* vector1, float* vector2, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numElements; i += stride) {
        result[i] = vector1[i] * vector2[i];
    }
}

__global__
void ReductionKernel(int numElements, float* input, float* result) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numElements) {
        sdata[tid] = input[index];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}