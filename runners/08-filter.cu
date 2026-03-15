#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "Filter.cuh"

int main() {
    std::ofstream results("benchmark_filter.csv");
    results << "N,BlockSize,TimeMs\n";

    const int MIN_LOG = 10;
    const int MAX_LOG = 26;
    const int MIN_BLOCK = 64;
    const int MAX_BLOCK = 1024;
    const int WARMUP_RUNS = 5;
    const int REPEAT_RUNS = 10;

    for (int logN = MIN_LOG; logN <= MAX_LOG; ++logN) {
        int N = 1 << logN;
        size_t size = N * sizeof(float);

        float* h_array = (float*)malloc(size);
        for (int i = 0; i < N; ++i) {
            h_array[i] = (float)(i % 100) / 100.0f;
        }

        float h_value = 0.5f;

        float* d_array;
        float* d_value;
        float* d_result;
        float* d_aux1;
        float* d_aux2;

        cudaMalloc(&d_array, size);
        cudaMalloc(&d_value, sizeof(float));
        cudaMalloc(&d_result, size);
        cudaMalloc(&d_aux1, sizeof(int));
        cudaMalloc(&d_aux2, size);

        cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, &h_value, sizeof(float), cudaMemcpyHostToDevice);

        for (int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize <<= 1) {
            int gridSize = (N + blockSize - 1) / blockSize;

            for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
                cudaMemset(d_aux1, 0, sizeof(int));
                Filter<<<gridSize, blockSize>>>(
                    N,
                    d_array,
                    GT,
                    d_value,
                    d_result,
                    d_aux1,
                    d_aux2
                );
            }
            cudaDeviceSynchronize();

            float totalTime = 0.0f;

            for (int repeat = 0; repeat < REPEAT_RUNS; ++repeat) {
                cudaMemset(d_aux1, 0, sizeof(int));

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start, 0);
                Filter<<<gridSize, blockSize>>>(
                    N,
                    d_array,
                    GT,
                    d_value,
                    d_result,
                    d_aux1,
                    d_aux2
                );
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);

                float elapsedTime = 0.0f;
                cudaEventElapsedTime(&elapsedTime, start, stop);
                totalTime += elapsedTime;

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }

            float averageTime = totalTime / REPEAT_RUNS;
            results << N << "," << blockSize << "," << averageTime << "\n";
        }

        cudaFree(d_array);
        cudaFree(d_value);
        cudaFree(d_result);
        cudaFree(d_aux1);
        cudaFree(d_aux2);

        free(h_array);
    }

    results.close();
    return 0;
}