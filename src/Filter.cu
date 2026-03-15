#include "Filter.cuh"

__global__ void Filter(
    int numElements,
    float* array,
    OperationFilterType type,
    float* value,
    float* result,
    float* auxArray1,
    float* auxArray2
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int* counter = (int*)auxArray1;

    for (int i = index; i < numElements; i += stride) {
        bool condition = false;

        if (type == GT) {
            condition = array[i] > value[0];
        } else if (type == LT) {
            condition = array[i] < value[0];
        }

        if (condition) {
            int pos = atomicAdd(counter, 1);
            result[pos] = array[i];
        }
    }
}