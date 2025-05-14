#include <sycl/sycl.hpp>
#include "sycl_matrix_multiplication.hpp"

struct matrix_mult;

// Updated to handle non-square matrices:
// A is (rowsA x colsA), B is (rowsB x colsB), result C is (rowsA x colsB)
// Note: colsA == rowsB must hold for multiplication
void sycl_matrix_multiply(const float* A, const float* B, float* C,
                          size_t rowsA, size_t colsA, size_t colsB) {
    // ✅ Use SYCL 2020 style: callable selector + enable profiling
    sycl::queue queue(sycl::default_selector_v,
                      sycl::property::queue::enable_profiling{});

    // Print the selected SYCL device
    std::cout << "SYCL is running on device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Start profiling
    sycl::event kernel_event;

    {
        sycl::buffer<float, 2> bufferA(A, sycl::range<2>(rowsA, colsA));
        sycl::buffer<float, 2> bufferB(B, sycl::range<2>(colsA, colsB));
        sycl::buffer<float, 2> bufferC(C, sycl::range<2>(rowsA, colsB));

        kernel_event = queue.submit([&](sycl::handler& cgh) {
            auto accA = bufferA.get_access<sycl::access::mode::read>(cgh);
            auto accB = bufferB.get_access<sycl::access::mode::read>(cgh);
            auto accC = bufferC.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<matrix_mult>(
                sycl::range<2>(rowsA, colsB),
                [=](sycl::item<2> item) {
                    size_t row = item.get_id(0);
                    size_t col = item.get_id(1);
                    float sum = 0.0f;

                    for (size_t k = 0; k < colsA; ++k) {
                        sum += accA[row][k] * accB[k][col];
                    }
                    accC[row][col] = sum;
                });
        });

        kernel_event.wait_and_throw();
    }

    // ✅ Collect kernel execution time
    auto start = kernel_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end   = kernel_event.get_profiling_info<sycl::info::event_profiling::command_end>();

    double duration_ms = static_cast<double>(end - start) * 1e-6; // ns to ms
    std::cout << "SYCL kernel execution time: " << duration_ms << " ms" << std::endl;
}