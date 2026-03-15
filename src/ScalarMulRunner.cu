#include "ScalarMulRunner.cuh"
#include "ScalarMul.cuh"

__global__
void MultiplyVectorsKernel(int numElements, float* vector1, float* vector2, float* result);

__global__
void ReductionKernel(int numElements, float* input, float* result);

float ScalarMulTwoReductions(int numElements, float* vector1, float* vector2, int blockSize) {
    float* d_vector1;
    float* d_vector2;
    float* d_products;
    float* d_partial1;
    float* d_partial2;

    cudaMalloc(&d_vector1, sizeof(float) * numElements);
    cudaMalloc(&d_vector2, sizeof(float) * numElements);
    cudaMalloc(&d_products, sizeof(float) * numElements);

    cudaMemcpy(d_vector1, vector1, sizeof(float) * numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, vector2, sizeof(float) * numElements, cudaMemcpyHostToDevice);

    int gridSize1 = (numElements + blockSize - 1) / blockSize;
    cudaMalloc(&d_partial1, sizeof(float) * gridSize1);

    MultiplyVectorsKernel<<<gridSize1, blockSize>>>(numElements, d_vector1, d_vector2, d_products);

    ReductionKernel<<<gridSize1, blockSize, sizeof(float) * blockSize>>>(numElements, d_products, d_partial1);

    int gridSize2 = (gridSize1 + blockSize - 1) / blockSize;
    cudaMalloc(&d_partial2, sizeof(float) * gridSize2);

    ReductionKernel<<<gridSize2, blockSize, sizeof(float) * blockSize>>>(gridSize1, d_partial1, d_partial2);

    float* h_partial = new float[gridSize2];
    cudaMemcpy(h_partial, d_partial2, sizeof(float) * gridSize2, cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < gridSize2; ++i) {
        result += h_partial[i];
    }

    delete[] h_partial;

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_products);
    cudaFree(d_partial1);
    cudaFree(d_partial2);

    return result;
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
    float* d_vector1;
    float* d_vector2;
    float* d_input;
    float* d_output;

    cudaMalloc(&d_vector1, sizeof(float) * numElements);
    cudaMalloc(&d_vector2, sizeof(float) * numElements);

    cudaMemcpy(d_vector1, vector1, sizeof(float) * numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, vector2, sizeof(float) * numElements, cudaMemcpyHostToDevice);

    int currentSize = numElements;
    int gridSize = (currentSize + blockSize - 1) / blockSize;

    cudaMalloc(&d_input, sizeof(float) * gridSize);

    ScalarMulBlock<<<gridSize, blockSize, sizeof(float) * blockSize>>>(currentSize, d_vector1, d_vector2, d_input);

    currentSize = gridSize;

    while (currentSize > 1) {
        int nextGridSize = (currentSize + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, sizeof(float) * nextGridSize);

        ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSize, d_input, d_output);

        cudaFree(d_input);
        d_input = d_output;
        currentSize = nextGridSize;
    }

    float result = 0.0f;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_input);

    return result;
}