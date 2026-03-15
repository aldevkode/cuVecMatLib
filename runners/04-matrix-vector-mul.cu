#include <fstream>
#include <cuda_runtime.h>
#include "MatrixVectorMul.cuh"


void FillMatrix(float* matrix, int height, int width, float value) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i * width + j] = value;
        }
    }
}

void FillVector(float* vector, int size, float value) {
    for (int i = 0; i < size; ++i) {
        vector[i] = value;
    }
}

int main() {
    std::ofstream results("benchmark_task_matrix_vector_mul.csv");
    results << "Height,Width,BlockSize,TimeMs\n";

    const int MIN_LOG = 5;
    const int MAX_LOG = 11;
    const int MIN_BLOCK = 64;
    const int MAX_BLOCK = 1024;
    const int WARMUP_RUNS = 5;
    const int REPEAT_RUNS = 10;

    for (int logH = MIN_LOG; logH <= MAX_LOG; ++logH) {
        for (int logW = MIN_LOG; logW <= MAX_LOG; ++logW) {
            int height = 1 << logH;
            int width = 1 << logW;

            float* h_matrix = new float[height * width];
            float* h_vector = new float[width];
            float* h_result = new float[height];

            FillMatrix(h_matrix, height, width, 1.0f);
            FillVector(h_vector, width, 2.0f);

            float* d_matrix;
            float* d_vector;
            float* d_result;

            cudaMalloc(&d_matrix, sizeof(float) * height * width);
            cudaMalloc(&d_vector, sizeof(float) * width);
            cudaMalloc(&d_result, sizeof(float) * height);

            cudaMemcpy(d_matrix, h_matrix, sizeof(float) * height * width, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vector, h_vector, sizeof(float) * width, cudaMemcpyHostToDevice);

            for (int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize <<= 1) {
                int gridSize = (height + blockSize - 1) / blockSize;

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
                    MatrixVectorMul<<<gridSize, blockSize>>>(height, width, d_matrix, d_vector, d_result);
                }
                cudaDeviceSynchronize();

                cudaEventRecord(start, 0);
                for (int repeat = 0; repeat < REPEAT_RUNS; ++repeat) {
                    MatrixVectorMul<<<gridSize, blockSize>>>(height, width, d_matrix, d_vector, d_result);
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);

                float elapsedTime;
                cudaEventElapsedTime(&elapsedTime, start, stop);
                elapsedTime /= REPEAT_RUNS;

                results << height << "," << width << "," << blockSize << "," << elapsedTime << "\n";

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }

            cudaMemcpy(h_result, d_result, sizeof(float) * height, cudaMemcpyDeviceToHost);

            cudaFree(d_matrix);
            cudaFree(d_vector);
            cudaFree(d_result);

            delete[] h_matrix;
            delete[] h_vector;
            delete[] h_result;
        }
    }

    results.close();
    return 0;
}