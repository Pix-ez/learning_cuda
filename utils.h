#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>



// Function to measure execution time
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_nsec + ts.tv_sec * 1e-9;

}

#define SEED  42

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
