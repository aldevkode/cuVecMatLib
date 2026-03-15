#include "MatrixMul.cuh"
#include <cuda_runtime.h>

__global__
void MatrixMul(int heightA, int widthA, int widthB, float* matrixA, float* matrixB, float* matrixResult) {
    extern __shared__ float shared[];

    int blockSize = blockDim.x;

    float* tileA = shared;
    float* tileB = shared + blockSize * blockSize;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockSize + ty;
    int col = blockIdx.x * blockSize + tx;

    float sum = 0.0f;

    int numTiles = (widthA + blockSize - 1) / blockSize;

    for (int tile = 0; tile < numTiles; ++tile) {
        int aCol = tile * blockSize + tx;
        int bRow = tile * blockSize + ty;

        if (row < heightA && aCol < widthA) {
            tileA[ty * blockSize + tx] = matrixA[row * widthA + aCol];
        } else {
            tileA[ty * blockSize + tx] = 0.0f;
        }

        if (bRow < widthA && col < widthB) {
            tileB[ty * blockSize + tx] = matrixB[bRow * widthB + col];
        } else {
            tileB[ty * blockSize + tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < blockSize; ++k) {
            sum += tileA[ty * blockSize + k] * tileB[k * blockSize + tx];
        }

        __syncthreads();
    }

    if (row < heightA && col < widthB) {
        matrixResult[row * widthB + col] = sum;
    }
}