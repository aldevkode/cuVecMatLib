#pragma once


/*
 * Calculates scalar multiplication for block
 */
__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float *result);

__global__
void MultiplyVectorsKernel(int numElements, float* vector1, float* vector2, float* result);

__global__
void ReductionKernel(int numElements, float* input, float* result);