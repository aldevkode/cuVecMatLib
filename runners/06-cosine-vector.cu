#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "ScalarMul.cuh"

__global__
void MultiplyVectorsKernel(int numElements, float* vector1, float* vector2, float* result);

__global__
void ReductionKernel(int numElements, float* input, float* result);

int main() {
    std::ofstream results("benchmark_cosine_vector.csv");
    results << "N,BlockSize,TimeMs\n";

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
            h_y[i] = 1.0f;
        }

        float* d_x;
        float* d_y;

        cudaMalloc(&d_x, sizeof(float) * N);
        cudaMalloc(&d_y, sizeof(float) * N);

        cudaMemcpy(d_x, h_x, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, sizeof(float) * N, cudaMemcpyHostToDevice);

        for (int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize <<= 1) {
            int firstGridSize = (N + blockSize - 1) / blockSize;

            float* d_xy_stage1;
            float* d_xy_stage2;
            float* d_xx_stage1;
            float* d_xx_stage2;
            float* d_yy_stage1;
            float* d_yy_stage2;

            cudaMalloc(&d_xy_stage1, sizeof(float) * firstGridSize);
            cudaMalloc(&d_xy_stage2, sizeof(float) * firstGridSize);
            cudaMalloc(&d_xx_stage1, sizeof(float) * firstGridSize);
            cudaMalloc(&d_xx_stage2, sizeof(float) * firstGridSize);
            cudaMalloc(&d_yy_stage1, sizeof(float) * firstGridSize);
            cudaMalloc(&d_yy_stage2, sizeof(float) * firstGridSize);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
                ScalarMulBlock<<<firstGridSize, blockSize, sizeof(float) * blockSize>>>(N, d_x, d_y, d_xy_stage1);
                ScalarMulBlock<<<firstGridSize, blockSize, sizeof(float) * blockSize>>>(N, d_x, d_x, d_xx_stage1);
                ScalarMulBlock<<<firstGridSize, blockSize, sizeof(float) * blockSize>>>(N, d_y, d_y, d_yy_stage1);

                int currentSizeXY = firstGridSize;
                int currentSizeXX = firstGridSize;
                int currentSizeYY = firstGridSize;

                float* d_inputXY = d_xy_stage1;
                float* d_outputXY = d_xy_stage2;

                float* d_inputXX = d_xx_stage1;
                float* d_outputXX = d_xx_stage2;

                float* d_inputYY = d_yy_stage1;
                float* d_outputYY = d_yy_stage2;

                while (currentSizeXY > 1) {
                    int nextGridSize = (currentSizeXY + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSizeXY, d_inputXY, d_outputXY);
                    currentSizeXY = nextGridSize;
                    float* temp = d_inputXY;
                    d_inputXY = d_outputXY;
                    d_outputXY = temp;
                }

                while (currentSizeXX > 1) {
                    int nextGridSize = (currentSizeXX + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSizeXX, d_inputXX, d_outputXX);
                    currentSizeXX = nextGridSize;
                    float* temp = d_inputXX;
                    d_inputXX = d_outputXX;
                    d_outputXX = temp;
                }

                while (currentSizeYY > 1) {
                    int nextGridSize = (currentSizeYY + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSizeYY, d_inputYY, d_outputYY);
                    currentSizeYY = nextGridSize;
                    float* temp = d_inputYY;
                    d_inputYY = d_outputYY;
                    d_outputYY = temp;
                }
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start, 0);
            for (int repeat = 0; repeat < REPEAT_RUNS; ++repeat) {
                ScalarMulBlock<<<firstGridSize, blockSize, sizeof(float) * blockSize>>>(N, d_x, d_y, d_xy_stage1);
                ScalarMulBlock<<<firstGridSize, blockSize, sizeof(float) * blockSize>>>(N, d_x, d_x, d_xx_stage1);
                ScalarMulBlock<<<firstGridSize, blockSize, sizeof(float) * blockSize>>>(N, d_y, d_y, d_yy_stage1);

                int currentSizeXY = firstGridSize;
                int currentSizeXX = firstGridSize;
                int currentSizeYY = firstGridSize;

                float* d_inputXY = d_xy_stage1;
                float* d_outputXY = d_xy_stage2;

                float* d_inputXX = d_xx_stage1;
                float* d_outputXX = d_xx_stage2;

                float* d_inputYY = d_yy_stage1;
                float* d_outputYY = d_yy_stage2;

                while (currentSizeXY > 1) {
                    int nextGridSize = (currentSizeXY + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSizeXY, d_inputXY, d_outputXY);
                    currentSizeXY = nextGridSize;
                    float* temp = d_inputXY;
                    d_inputXY = d_outputXY;
                    d_outputXY = temp;
                }

                while (currentSizeXX > 1) {
                    int nextGridSize = (currentSizeXX + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSizeXX, d_inputXX, d_outputXX);
                    currentSizeXX = nextGridSize;
                    float* temp = d_inputXX;
                    d_inputXX = d_outputXX;
                    d_outputXX = temp;
                }

                while (currentSizeYY > 1) {
                    int nextGridSize = (currentSizeYY + blockSize - 1) / blockSize;
                    ReductionKernel<<<nextGridSize, blockSize, sizeof(float) * blockSize>>>(currentSizeYY, d_inputYY, d_outputYY);
                    currentSizeYY = nextGridSize;
                    float* temp = d_inputYY;
                    d_inputYY = d_outputYY;
                    d_outputYY = temp;
                }
            }
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsedTime = 0.0f;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            elapsedTime /= REPEAT_RUNS;

            results << N << "," << blockSize << "," << elapsedTime << "\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaFree(d_xy_stage1);
            cudaFree(d_xy_stage2);
            cudaFree(d_xx_stage1);
            cudaFree(d_xx_stage2);
            cudaFree(d_yy_stage1);
            cudaFree(d_yy_stage2);
        }

        cudaFree(d_x);
        cudaFree(d_y);

        free(h_x);
        free(h_y);
    }

    results.close();
    return 0;
}