#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "config.h"
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <iomanip>
#include <iomanip>
#include <functional>
#include <numeric>


// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + (ts.tv_nsec * 1e-9); // Properly convert ns to seconds
}



void set_seed(){
    srand(SEED);
}
    
//Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols){
    for (int i=0; i < rows*cols; i++ ){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

//CPU matmul
void matmul_cpu(float *A, float *B, float *C, int m, int n, int k){
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            float sum = 0.0f;
            for (int l=0; l<k; l++){
                sum += A[i * k + l] * B [l *n +j];

            }
            C[i* n + j] =sum;
        }
    }
}

//Compare results
void compare_results(float *A, float *B, int rows, int cols){
    bool match = true;
    for (int i = 0; i < rows * cols; ++i){
        float diff = std::abs(A[i] - B[i]);
        if (diff > 1e-4f){
            std::cout << "Mismatch at index " << i << ": A = "
                      << A[i] << ", B = " << B[i] << ", diff = " << diff << "\n";
            match = false;
            break;
        }
    }
    std::cout << "Results " << (match ? "matched ✅" : "did NOT match ❌") << std::endl;
}

// bool verifyResultsVec(const std::vector<float>& expected, const std::vector<float>& actual, float tolerance = 1e-2) {
//     if (expected.size() != actual.size()) {
//         return false;
//     }

//     for (size_t i = 0; i < expected.size(); ++i) {
//         float rel_error = std::abs(expected[i] - actual[i]);
//         if (rel_error > tolerance) {
//             std::cout << "Mismatch at index " << i << ": expected " << expected[i] 
//                       << ", got " << actual[i] << ", relative error: " << rel_error << std::endl;
//             return false;
//         }
//     }

//     return true;
// }


// Assuming you can modify the function definition:
bool verifyResults(const float* actual, const float* expected, float tolerance, int size) {
    for (int i = 0; i < size; ++i) {
        float rel_error = std::abs(expected[i] - actual[i]);
        if (rel_error > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << actual[i] << ", relative error: " << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

// New function for CUDA event-based timing
float time_kernel(std::function<void()> kernel_func) {
    cudaEvent_t start, stop;
    float elapsed_time;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel_func();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}

// Function to perform warmup and benchmark runs
float benchmark_kernel(std::function<void()> kernel_func, int warmup_runs, int benchmark_runs) {
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        kernel_func();
    }
    
    // Benchmark runs
    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i) {
        float time = time_kernel(kernel_func);
        times.push_back(time);
    }
    
    // Calculate average time
    float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;
    return avg_time;
}
