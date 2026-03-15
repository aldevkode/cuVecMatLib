#include <cmath>
#include "CosineVector.cuh"
#include "ScalarMulRunner.cuh"

float CosineVector(int numElements, float* vector1, float* vector2, int blockSize) {
    float ab = ScalarMulSumPlusReduction(numElements, vector1, vector2, blockSize);
    float aa = ScalarMulSumPlusReduction(numElements, vector1, vector1, blockSize);
    float bb = ScalarMulSumPlusReduction(numElements, vector2, vector2, blockSize);

    float norm1 = std::sqrt(aa);
    float norm2 = std::sqrt(bb);

    if (norm1 == 0.0f || norm2 == 0.0f) {
        return 0.0f;
    }

    return ab / (norm1 * norm2);
}