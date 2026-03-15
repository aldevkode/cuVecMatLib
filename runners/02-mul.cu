#include <fstream>
#include <cuda_runtime.h>
#include "KernelMul.cuh"


int main() {
    cudaSetDevice(5);

    std::ofstream results("benchmark_task_mul.csv");
    results << "N,BlockSize,TimeMs\n";

    const int MIN_LOG = 10;
    const int MAX_LOG = 26;
    const int MIN_BLOCK = 64;
    const int MAX_BLOCK = 1024;

    for (int logN = MIN_LOG; logN <= MAX_LOG; ++logN) {
        int N = 1 << logN;
        size_t size = N * sizeof(float);

        float* h_x = (float*)malloc(size);
        float* h_y = (float*)malloc(size);
        float* d_x;
        float* d_y;
        float* d_result;

        cudaMalloc(&d_x, size);
        cudaMalloc(&d_y, size);
        cudaMalloc(&d_result, size);

        for (int i = 0; i < N; ++i) {
            h_x[i] = 2.0f;
            h_y[i] = 3.0f;
        }

        cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

        for (int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize <<= 1) {
            int gridSize = (N + blockSize - 1) / blockSize;

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            KernelMul<<<gridSize, blockSize>>>(N, d_x, d_y, d_result);
            cudaDeviceSynchronize();

            cudaEventRecord(start, 0);
            KernelMul<<<gridSize, blockSize>>>(N, d_x, d_y, d_result);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);

            results << N << "," << blockSize << "," << elapsedTime << "\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_result);
        free(h_x);
        free(h_y);
    }

    results.close();
    return 0;
}