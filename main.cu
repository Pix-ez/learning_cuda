#include <iostream>
#include <cuda_runtime.h>
#include "utils.h"
#include "config.h"


// Kernel declaration
__global__ void naiveMatrixMultiply(float *A, float *B, float *C, int M, int N, int K);
__global__ void tiledMatrixMultiply(float *A, float *B, float *C, int M, int N, int K);

int main(){

    //Matrix: C = A x B --> [M x K] * [K * N] = [M x N]
    int M = 8192;
    int K = 1024;
    int N = 4096;

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
    dim3 default_blockDim(16, 16);
    dim3 default_gridDim((N + default_blockDim.x - 1) / default_blockDim.x,
                         (M + default_blockDim.y - 1) / default_blockDim.y);
   

    int iterations = 20;

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    // for (int i = 0; i < 3; i++) {
    //     matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    //     naiveMatrixMultiply<<<default_gridDim, default_blockDim>>>(d_A, d_B, d_C, M, N, K);
    //     cudaDeviceSynchronize();
    // }

 



    //############################################## Naive MM ##############################################
    // double gpu_total_time = 0.0;
    // Kernel launch config

    // for (int i=0; i<iterations; ++i){
    //     double start_time = get_time();
    //     naiveMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    //     cudaDeviceSynchronize();  // wait for kernel to finish to get correct time
    //     double end_time = get_time();
    //     gpu_total_time += end_time - start_time;
    // }

    // double gpu_avg_time = gpu_total_time / double(iterations);
    // printf("Naive MM average time: %f microseconds\n", (gpu_avg_time * time_scale));

    // cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    // compare_results(h_C, h_C_cpu, M, N);




    //############################################## Tiled kernel ##############################################
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    
    double gpu_total_time = 0.0;
    for (int i=0; i<iterations; ++i){
        double start_time = get_time();
        tiledMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();  // wait for kernel to finish to get correct time
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }

    double gpu_avg_time = gpu_total_time / double(iterations);
    
    // std::cout << "CPU Time: " << cpu_avg_time * time_scale << " ms" << std::endl;
    printf("Tiled MM average time: %f microseconds\n", (gpu_avg_time * time_scale));

    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);


    // // CPU reference
    // double cpu_total_time = 0.0;
    // for (int i=0; i<iterations; ++i){
    //     double start_time = get_time();
    //     matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    //     double end_time = get_time();
    //     cpu_total_time += end_time - start_time;
    // }

    // double cpu_avg_time = cpu_total_time / double(iterations);
    
    // printf("CPU average time: %f microseconds\n", (cpu_avg_time * time_scale));

    // // Verify correctness
    // compare_results(h_C, h_C_cpu, M, N);






    // Free memory
    free(h_A); free(h_B); free(h_C); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;


}