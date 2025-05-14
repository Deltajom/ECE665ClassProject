# ECE665ClassProject

This project implements matrix multiplication using both the raw C++ standard library and SYCL for acceleration. It serves as a demonstration of matrix operations and the benefits of parallel computing.

For a system with...
CPU - 12900K
RAM - 32 GB DDR5
GPU - NVIDIA RTX 3090

the program showed as follows...

Benchmarking matrix multiplication: 2048x2048 * 2048x2048
[CPU] multiply took: 6518 ms
SYCL is running on device: NVIDIA GeForce RTX 3090
SYCL kernel execution time: 15.8628 ms
[Check] SYCL C[0][0] = 41252
Done.

## Project Structure

```
ECE665ClassProject
├── src
│   ├── main.cpp
│   ├── matrix_multiplication.cpp
│   └── sycl_matrix_multiplication.cpp
├── include
|   ├── sycl_matrix_multiplication.hpp
|   ├── matrix_multiplication.hpp
│   └── utils.hpp
├── CMakeLists.txt
└── README.md
```

## Files Overview

- **src/main.cpp**: Entry point of the application
  
- **src/matrix_multiplication.cpp**: Implements matrix multiplication using the C++ standard library. Contains functions for multiplying two matrices and returning the result.
  
- **include/matrix_multiplication.hpp**: Header file that declares the functions for matrix multiplication using the standard library.
  
- **src/sycl_matrix_multiplication.cpp**: Implements matrix multiplication using SYCL for acceleration. Defines the SYCL kernel and the host code that sets up the data and invokes the kernel.
  
- **include/sycl_matrix_multiplication.hpp**: Header file that declares the functions for matrix multiplication using SYCL.
  
- **include/utils.hpp**: Contains utility functions for generating random matrices or printing matrices.

- **CMakeLists.txt**: Configuration file for CMake, specifying the project name, required C++ standard, and includes the source and header files for building the project.

## Building the Project

Built under Ubuntu 22.04, should work on 24.04 as well but untested

To build the project, follow these steps:

1. Ensure you have CMake installed on your system.
2. Install a working nvidia driver if using GPU, and relative cuda install as well (tested on 12.4)
3. Run the install `installprerequisities.sh` script
4. Download the DPC++ compiler (now unified under icpx) and setup with the installer type of your choice here - https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html?operatingsystem=linux&distribution-linux=offline
5. Download and setup the latest oneAPI for nvidia GPUs from here - https://developer.codeplay.com/products/oneapi/nvidia/download/
6. Run `source /opt/intel/oneapi/setvars.sh`
7. Test to see if installation was successful `sycl-ls`
8. Open a terminal and navigate to the project directory.
9. Create a build directory:
   ```
   mkdir build
   cd build
   ```
10. Run CMake to configure the project:
   ```
   cmake ..
   ```
11. Build the project:
   ```
   make
   ```

Note - you can put `source /opt/intel/oneapi/setvars.sh` in your bashrc file if you are going to use sycl in more projects

## Running the Project

After building the project, you can run the executable generated in the `build` directory. The program will multiply 2 large matricies, and time the execution on cpu using the regular standard library, and on the first available system-wide sycl compatible accelerator.

## Matrix Multiplication Implementations

This project showcases two implementations of matrix multiplication:

1. **Standard C++ Implementation**: Utilizes the standard library for matrix operations.
2. **SYCL Implementation**: Leverages SYCL for parallel execution, allowing for potentially faster computations on compatible hardware.

Please cite this example if used!
