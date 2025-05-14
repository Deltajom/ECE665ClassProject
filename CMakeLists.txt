cmake_minimum_required(VERSION 3.20.5)
project(MySYCLProject LANGUAGES CXX)

# Set the DPC++ compiler
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O2")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add your source files
add_executable(my_sycl_app
    "src/main.cpp"
    "src/sycl_matrix_multiplication.cpp"
    "src/matrix_multiplication.cpp"
)

# Specify SYCL and CUDA compilation options
target_compile_options(my_sycl_app PRIVATE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86
)

target_link_options(my_sycl_app PRIVATE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86
)

# Link against the SYCL runtime library
target_link_libraries(my_sycl_app PRIVATE sycl cuda)