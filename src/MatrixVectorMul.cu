#include <MatrixVectorMul.cuh>

__global__ void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = i; row < height; row += stride) {
        float sum = 0.0f;
        for (int j = 0; j < width; ++j) {
            sum += matrix[row * width + j] * vector[j];
        }
        result[row] = sum;
    }
}

