#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "ScalarMul.cuh"

__global__
void MultiplyVectorsKernel(int numElements, float* vector1, float* vector2, float* result);

__global__
void ReductionKernel(int numElements, float* input, float* result);

int main() {
    std::ofstream results1("benchmark_scalar_mul_two_reductions.csv");
    std::ofstream results2("benchmark_scalar_mul_sum_plus_reduction.csv");

    results1 << "N,BlockSize,TimeMs\n";
    results2 << "N,BlockSize,TimeMs\n";

    const int MIN_LOG = 10;
    const int MAX_LOG = 26;
    const int MIN_BLOCK = 64;
    const int MAX_BLOCK = 1024;
    const int WARMUP_RUNS = 5;
    const int REPEAT_RUNS = 10;

    for (int logN = MIN_LOG; logN <= MAX_LOG; ++logN) {
        int N = 1 << logN;

        float* h_x = (float*)malloc(sizeof(float) * N);
        float* h_y = (float*)malloc(sizeof(float) * N);

        for (int i = 0; i < N; ++i) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
        }

        float* d_x;
        float* d_y;
        float* d_products;
        float* d_partial1;
        float* d_partial2;
        float* d_partialA;
        float* d_partialB;

        cudaMalloc(&d_x, sizeof(float) * N);
        cudaMalloc(&d_y, sizeof(float) * N);
        cudaMalloc(&d_products, sizeof(float) * N);

        cudaMemcpy(d_x, h_x, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, sizeof(float) * N, cudaMemcpyHostToDevice);

        for (int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize <<= 1) {
            int gridSize1 = (N + blockSize - 1) / blockSize;
            int gridSize2 = (gridSize1 + blockSize - 1) / blockSize;

            cudaMalloc(&d_partial1, sizeof(float) * gridSize1);
            cudaMalloc(&d_partial2, sizeof(float) * gridSize2);
            cudaMalloc(&d_partialA, sizeof(float) * gridSize1);
            cudaMalloc(&d_partialB, sizeof(float) * gridSize1);

            cudaEvent_t start, stop;
            float elapsedTime;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
                MultiplyVectorsKernel<<<gridSize1, blockSize>>>(N, d_x, d_y, d_products);
                ReductionKernel<<<gridSize1, blockSize, sizeof(float) * blockSize>>>(N, d_products, d_partial1);
                ReductionKernel<<<gridSize2, blockSize, sizeof(float) * blockSize>>>(gridSize1, d_partial1, d_partial2);
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start, 0);
            for (int repeat = 0; repeat < REPEAT_RUNS; ++repeat) {
                MultiplyVectorsKernel<<<gridSize1, blockSize>>>(N, d_x, d_y, d_products);
                ReductionKernel<<<gridSize1, blockSize, sizeof(float) * blockSize>>>(N, d_products, d_partial1);
                ReductionKernel<<<gridSize2, blockSize, sizeof(float) * blockSize>>>(gridSize1, d_partial1, d_partial2);
            }
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&elapsedTime, start, stop);
            elapsedTime /= REPEAT_RUNS;

            results1 << N << "," << blockSize << "," << elapsedTime << "\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
                ScalarMulBlock<<<gridSize1, blockSize, sizeof(float) * blockSize>>>(N, d_x, d_y, d_partialA);

                int currentSize = gridSize1;
                float* d_input = d_partialA;
                float* d_output = d_partialB;

                while (currentSize > 1) {
                    int nextGridSize = (currentSize + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSize, d_input, d_output);

                    currentSize = nextGridSize;

                    float* temp = d_input;
                    d_input = d_output;
                    d_output = temp;
                }
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start, 0);
            for (int repeat = 0; repeat < REPEAT_RUNS; ++repeat) {
                ScalarMulBlock<<<gridSize1, blockSize, sizeof(float) * blockSize>>>(N, d_x, d_y, d_partialA);

                int currentSize = gridSize1;
                float* d_input = d_partialA;
                float* d_output = d_partialB;

                while (currentSize > 1) {
                    int nextGridSize = (currentSize + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSize, d_input, d_output);

                    currentSize = nextGridSize;

                    float* temp = d_input;
                    d_input = d_output;
                    d_output = temp;
                }
            }
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&elapsedTime, start, stop);
            elapsedTime /= REPEAT_RUNS;

            results2 << N << "," << blockSize << "," << elapsedTime << "\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaFree(d_partial1);
            cudaFree(d_partial2);
            cudaFree(d_partialA);
            cudaFree(d_partialB);
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_products);

        free(h_x);
        free(h_y);
    }

    results1.close();
    results2.close();

    return 0;
}