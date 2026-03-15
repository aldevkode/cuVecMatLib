#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "MatrixMul.cuh"

void FillMatrix(float* matrix, int height, int width, float value) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i * width + j] = value;
        }
    }
}

int main() {
    std::ofstream results("benchmark_matrix_mul.csv");
    results << "HeightA,WidthA,WidthB,BlockSize,TimeMs\n";

    const int MIN_LOG = 5;
    const int MAX_LOG = 10;
    const int MIN_BLOCK = 8;
    const int MAX_BLOCK = 32;
    const int WARMUP_RUNS = 5;
    const int REPEAT_RUNS = 10;

    for (int logH = MIN_LOG; logH <= MAX_LOG; ++logH) {
        for (int logMid = MIN_LOG; logMid <= MAX_LOG; ++logMid) {
            for (int logW = MIN_LOG; logW <= MAX_LOG; ++logW) {
                int heightA = 1 << logH;
                int widthA = 1 << logMid;
                int widthB = 1 << logW;

                float* h_A = (float*)malloc(sizeof(float) * heightA * widthA);
                float* h_B = (float*)malloc(sizeof(float) * widthA * widthB);
                float* h_C = (float*)malloc(sizeof(float) * heightA * widthB);

                FillMatrix(h_A, heightA, widthA, 1.0f);
                FillMatrix(h_B, widthA, widthB, 2.0f);

                float* d_A;
                float* d_B;
                float* d_C;

                cudaMalloc(&d_A, sizeof(float) * heightA * widthA);
                cudaMalloc(&d_B, sizeof(float) * widthA * widthB);
                cudaMalloc(&d_C, sizeof(float) * heightA * widthB);

                cudaMemcpy(d_A, h_A, sizeof(float) * heightA * widthA, cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, h_B, sizeof(float) * widthA * widthB, cudaMemcpyHostToDevice);

                for (int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize <<= 1) {
                    dim3 block(blockSize, blockSize);
                    dim3 grid((widthB + blockSize - 1) / blockSize,
                              (heightA + blockSize - 1) / blockSize);

                    size_t sharedMemSize = 2 * blockSize * blockSize * sizeof(float);

                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
                        MatrixMul<<<grid, block, sharedMemSize>>>(heightA, widthA, widthB, d_A, d_B, d_C);
                    }
                    cudaDeviceSynchronize();

                    cudaEventRecord(start, 0);
                    for (int repeat = 0; repeat < REPEAT_RUNS; ++repeat) {
                        MatrixMul<<<grid, block, sharedMemSize>>>(heightA, widthA, widthB, d_A, d_B, d_C);
                    }
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);

                    float elapsedTime = 0.0f;
                    cudaEventElapsedTime(&elapsedTime, start, stop);
                    elapsedTime /= REPEAT_RUNS;

                    results << heightA << "," << widthA << "," << widthB << "," << blockSize << "," << elapsedTime << "\n";

                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);
                }

                cudaMemcpy(h_C, d_C, sizeof(float) * heightA * widthB, cudaMemcpyDeviceToHost);

                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);

                free(h_A);
                free(h_B);
                free(h_C);
            }
        }
    }

    results.close();
    return 0;
}