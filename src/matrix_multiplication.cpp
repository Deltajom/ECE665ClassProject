#include "matrix_multiplication.hpp"
#include <vector>
#include <stdexcept>

std::vector<std::vector<int>> multiply_matrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication.");
    }

    size_t rows_A = A.size();
    size_t cols_A = A[0].size();
    size_t cols_B = B[0].size();
    std::vector<std::vector<int>> C(rows_A, std::vector<int>(cols_B, 0));

    for (size_t i = 0; i < rows_A; ++i) {
        for (size_t j = 0; j < cols_B; ++j) {
            for (size_t k = 0; k < cols_A; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}