#ifndef SYCL_MATRIX_MULTIPLICATION_H
#define SYCL_MATRIX_MULTIPLICATION_H

#include <sycl/sycl.hpp>
#include <vector>

void sycl_matrix_multiply(const float* A, const float* B, float* C,
                          size_t rowsA, size_t colsA, size_t colsB);

#endif // SYCL_MATRIX_MULTIPLICATION_H