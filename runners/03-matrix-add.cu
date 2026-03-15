#include <fstream>
#include <cuda_runtime.h>
#include "KernelMatrixAdd.cuh"


void FillMatrix(float* matrix, int height, int width, float value) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i * width + j] = value;
        }
    }
}

int main() {
    std::ofstream results("benchmark_task_matrix_add.csv");
    results << "Height,Width,BlockX,BlockY,TimeMs\n";

    const int MIN_LOG = 5;
    const int MAX_LOG = 11;
    const int MIN_BLOCK = 8;
    const int MAX_BLOCK = 32;

    for (int logH = MIN_LOG; logH <= MAX_LOG; ++logH) {
        for (int logW = MIN_LOG; logW <= MAX_LOG; ++logW) {
            int height = 1 << logH;
            int width = 1 << logW;

            float* h_A = new float[height * width];
            float* h_B = new float[height * width];
            float* h_result = new float[height * width];

            FillMatrix(h_A, height, width, 1.0f);
            FillMatrix(h_B, height, width, 2.0f);

            float* d_A;
            float* d_B;
            float* d_result;

            size_t pitchBytesA;
            size_t pitchBytesB;
            size_t pitchBytesResult;

            cudaMallocPitch(&d_A, &pitchBytesA, width * sizeof(float), height);
            cudaMallocPitch(&d_B, &pitchBytesB, width * sizeof(float), height);
            cudaMallocPitch(&d_result, &pitchBytesResult, width * sizeof(float), height);

            cudaMemcpy2D(d_A, pitchBytesA,
                         h_A, width * sizeof(float),
                         width * sizeof(float), height,
                         cudaMemcpyHostToDevice);

            cudaMemcpy2D(d_B, pitchBytesB,
                         h_B, width * sizeof(float),
                         width * sizeof(float), height,
                         cudaMemcpyHostToDevice);

            int pitch = pitchBytesA / sizeof(float);

            for (int blockX = MIN_BLOCK; blockX <= MAX_BLOCK; blockX <<= 1) {
                for (int blockY = MIN_BLOCK; blockY <= MAX_BLOCK; blockY <<= 1) {
                    dim3 block(blockX, blockY);
                    dim3 grid((width + block.x - 1) / block.x,
                              (height + block.y - 1) / block.y);

                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    KernelMatrixAdd<<<grid, block>>>(height, width, pitch, d_A, d_B, d_result);
                    cudaDeviceSynchronize();

                    cudaEventRecord(start, 0);
                    KernelMatrixAdd<<<grid, block>>>(height, width, pitch, d_A, d_B, d_result);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);

                    float elapsedTime;
                    cudaEventElapsedTime(&elapsedTime, start, stop);

                    results << height << "," << width << "," << block.x << "," << block.y << "," << elapsedTime << "\n";

                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);
                }
            }

            cudaMemcpy2D(h_result, width * sizeof(float),
                         d_result, pitchBytesResult,
                         width * sizeof(float), height,
                         cudaMemcpyDeviceToHost);

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_result);

            delete[] h_A;
            delete[] h_B;
            delete[] h_result;
        }
    }

    results.close();
    return 0;
}