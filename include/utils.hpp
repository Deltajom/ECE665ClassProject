#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <random>

// Function to generate a random matrix of given dimensions
std::vector<std::vector<int>> generate_random_matrix(int rows, int cols);

// Function to print a matrix
void print_matrix(const std::vector<std::vector<int>>& matrix);

#endif // UTILS_H