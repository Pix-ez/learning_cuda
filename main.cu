#include <iostream>
#include <cuda_runtime.h>
#include "utils.h"


#define time_scale 1e6f
// Kernel declaration
__global__ void naiveMatrixMultiply(float *A, float *B, float *C, int M, int N, int K);


int main(){

    //Matrix: C = A x B --> [M x K] * [K * N] = [M x N]
    int M = 512;
    int K = 256;
    int N = 126;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    //Allocate host memory 
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);

    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    //kernel launch 
    // Kernel launch config
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    int iterations = 20;

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        naiveMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    // CPU reference
    double cpu_total_time = 0.0;
    for (int i=0; i<iterations; ++i){
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }

    double cpu_avg_time = cpu_total_time / double(iterations);
    
    // std::cout << "CPU Time: " << cpu_avg_time * time_scale << " ms" << std::endl;
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * time_scale));



    // Naive MM
    double gpu_total_time = 0.0;
    for (int i=0; i<iterations; ++i){
        double start_time = get_time();
        naiveMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();  // wait for kernel to finish to get correct time
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }

    double gpu_avg_time = gpu_total_time / double(iterations);
    
    // std::cout << "CPU Time: " << cpu_avg_time * time_scale << " ms" << std::endl;
    printf("Naive MM average time: %f microseconds\n", (gpu_avg_time * time_scale));

    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify correctness
    compare_results(h_C, h_C_cpu, M, N);

    // Free memory
    free(h_A); free(h_B); free(h_C); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;


}