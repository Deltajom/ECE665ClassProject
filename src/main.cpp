#include <iostream>
#include <vector>
#include <chrono>
#include "matrix_multiplication.hpp"
#include "sycl_matrix_multiplication.hpp"

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int rowsA = 2048;
    const int colsA = 2048;
    const int rowsB = colsA;
    const int colsB = 2048;

    std::cout << "Benchmarking matrix multiplication: " << rowsA << "x" << colsA
              << " * " << rowsB << "x" << colsB << std::endl;

    // Initialize matrices with random values
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));
    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB));

    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsA; ++j)
            A[i][j] = rand() % 10;

    for (int i = 0; i < rowsB; ++i)
        for (int j = 0; j < colsB; ++j)
            B[i][j] = rand() % 10;

    // ================== CPU Multiply ==================
    auto start_cpu = std::chrono::high_resolution_clock::now();
    result = multiply_matrices(A, B);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    std::cout << "[CPU] multiply took: " << duration_cpu << " ms\n";

    // ================== SYCL Multiply ==================
    std::vector<float> flatA(rowsA * colsA);
    std::vector<float> flatB(rowsB * colsB);
    std::vector<float> flatC(rowsA * colsB, 0.0f);

    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsA; ++j)
            flatA[i * colsA + j] = static_cast<float>(A[i][j]);

    for (int i = 0; i < rowsB; ++i)
        for (int j = 0; j < colsB; ++j)
            flatB[i * colsB + j] = static_cast<float>(B[i][j]);

    sycl_matrix_multiply(flatA.data(), flatB.data(), flatC.data(), rowsA, colsA, colsB);

    // Optional: verify a few elements
    std::cout << "[Check] SYCL C[0][0] = " << static_cast<int>(flatC[0]) << std::endl;
    std::cout << "Done." << std::endl;

    return 0;
}