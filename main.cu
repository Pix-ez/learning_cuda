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
    float *h_C_naive = (float*)malloc(size_C);
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

    // int iterations = 20;

    // Warm-up runs
    printf("Performing warm-up runs...\n");

    const int warmup_runs= 3;
    const int benchmark_runs= 20;



//############################################## Naive MM ##############################################

    // std::cout << h_A[1] << "\t"<< h_A[10] <<std::endl;

    dim3 default_blockDim(16, 16);
    dim3 default_gridDim((N + default_blockDim.x - 1) / default_blockDim.x,
                         (M + default_blockDim.y - 1) / default_blockDim.y);

    float naive_kernel = benchmark_kernel([&]() {
        naiveMatrixMultiply<<<default_gridDim, default_blockDim>>>(d_A, d_B, d_C, M, N, K);
    }, warmup_runs, benchmark_runs);
    std::cout << "Naive CUDA kernel average time: " << naive_kernel << " ms" << std::endl;

    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);




//############################################## Tiled kernel ##############################################

    // init_matrix(h_A, M, K);
    // init_matrix(h_B, K, N);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    
    float tiled_cuda_kernel = benchmark_kernel([&]() {
        tiledMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }, warmup_runs, benchmark_runs);
    std::cout << "Tiled CUDA kernel average time: " << tiled_cuda_kernel << " ms" << std::endl;

    // // Copy result back
    // cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);


//###################### CPU reference #####################################################
    // double cpu_total_time = 0.0;
    // for (int i=0; i<iterations; ++i){
    //     double start_time = get_time();
    //     matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    //     double end_time = get_time();
    //     cpu_total_time += end_time - start_time;
    // }

    // double cpu_avg_time = cpu_total_time / double(iterations);
    
    // printf("CPU average time: %f miliseconds\n", (cpu_avg_time * TIME_SCALE_MS));

    // // Verify correctness
    // compare_results(h_C, h_C_cpu, M, N);


//######################### CUBLASTLT FP32 #################################################
    //ALlocate Fp32 matrix
    float *h_C_cublaslt_fp32 = (float*)malloc(size_C);
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, size_A));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, size_B));
    CHECK_CUDA(cudaMalloc(&d_C_fp32, size_C));

    //Copy
    // init_matrix(h_A, M, K);
    // init_matrix(h_B, K, N);
    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B, size_B, cudaMemcpyHostToDevice));
    //CHECK_CUDA(cudaMemcpy(d_C_fp32, h_C_cublaslt_fp32, size_C, cudaMemcpyHostToDevice));
    // std::cout << h_A[1] << "\t"<< h_A[10] <<std::endl;
    //Create cublas handle
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    //set up matrix descriptors for FP32
    cublasLtMatrixLayout_t matA_fp32, matB_fp32, matC_fp32;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp32, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp32, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp32, CUDA_R_32F, N, M, N));

    //set up matrix multiplication descriptor for FP32
    cublasLtMatmulDesc_t matmulDesc_fp32;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    //set matrix operation for A & B
    cublasOperation_t transa = CUBLAS_OP_N; //this is 0 mean no transpose as we already made our matrix has same inner dim
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

    //setup alpha and beta
    const float alpha = 1.0f;
    const float beta =0.0f;


    float cublaslt_fp32_time = benchmark_kernel([&]() {
            CHECK_CUBLAS(cublasLtMatmul(
            handle,           // (1) Library context handle
            matmulDesc_fp32,  // (2) Operation descriptor
            &alpha,           // (3) Scalar multiplier for operation
            d_B_fp32,         // (4) Input matrix B pointer
            matB_fp32,        // (5) Layout descriptor for matrix B
            d_A_fp32,         // (6) Input matrix A pointer
            matA_fp32,        // (7) Layout descriptor for matrix A
            &beta,            // (8) Scalar multiplier for C
            d_C_fp32,         // (9) Input/output matrix C pointer
            matC_fp32,        // (10) Layout descriptor for C
            d_C_fp32,         // (11) Output matrix D pointer (same as C here)
            matC_fp32,        // (12) Layout descriptor for C
            nullptr,          // (13) Workspace pointer
            nullptr,          // (14) Preferences pointer
            0,                // (15) Workspace size
            0                 // (16) Stream ID
        ));

    }, warmup_runs, benchmark_runs);
    std::cout << "CublasLt FP32 kernel average time: " << cublaslt_fp32_time << " ms" << std::endl;

    
  
    cudaMemcpy(h_C_cublaslt_fp32, d_C_fp32, size_C, cudaMemcpyDeviceToHost);

    bool cublas_fp32_correct = verifyResults(h_C_naive, h_C_cublaslt_fp32, 1e-2, size_C);
    std::cout << "cuBLAS FP32 results " << (cublas_fp32_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 1e-2." << std::endl;

    free(h_C_cublaslt_fp32);
    cudaFree(d_A_fp32);cudaFree(d_B_fp32);cudaFree(d_C_fp32);


//######################### CUBLASTLT FP16 #################################################
    //ALlocate Fp16 matrix

    size_t half_size_A = M * K * sizeof(half);
    size_t half_size_B = K * N * sizeof(half);
    size_t half_size_C = M * N * sizeof(half);
    half *h_C_cublaslt_fp16 = (half*)malloc(half_size_C);
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc(&d_A_fp16, half_size_A));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, half_size_B));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, half_size_C));

    //first convert to FP16 then copy
    half *h_A_half = (half*)malloc(half_size_A);
    half *h_B_half = (half*)malloc(half_size_B);
    

    for (int i=0; i< M*K; ++i) h_A_half[i] = __float2half(h_A[i]);
    for (int i=0; i< N*K; ++i) h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A_half, half_size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B_half, half_size_B, cudaMemcpyHostToDevice));
    

    //Create cublas handle
    //cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    //set up matrix descriptors for FP16
    cublasLtMatrixLayout_t matA_fp16, matB_fp16, matC_fp16;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp16, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp16, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp16, CUDA_R_16F, N, M, N));

    //set up matrix multiplication descriptor for FP32
    cublasLtMatmulDesc_t matmulDesc_fp16;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    //set matrix operation for A & B
    //cublasOperation_t transa = CUBLAS_OP_N; //this is 0 mean no transpose as we already made our matrix has same inner dim
    //cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

    //setup alpha and beta
    half alpha_half = __float2half(1.0f);
    half beta_half = __float2half(0.0f);

    float cublaslt_fp16_time = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasLtMatmul(
            handle,           // (1) Library context handle
            matmulDesc_fp16,  // (2) Operation descriptor
            &alpha_half,           // (3) Scalar multiplier for operation
            d_B_fp16,         // (4) Input matrix B pointer
            matB_fp16,        // (5) Layout descriptor for matrix B
            d_A_fp16,         // (6) Input matrix A pointer
            matA_fp16,        // (7) Layout descriptor for matrix A
            &beta_half,            // (8) Scalar multiplier for C
            d_C_fp16,         // (9) Input/output matrix C pointer
            matC_fp16,        // (10) Layout descriptor for C
            d_C_fp16,         // (11) Output matrix D pointer (same as C here)
            matC_fp16,        // (12) Layout descriptor for C
            nullptr,          // (13) Workspace pointer
            nullptr,          // (14) Preferences pointer
            0,                // (15) Workspace size
            0                 // (16) Stream ID
        ));

    }, warmup_runs, benchmark_runs);
    std::cout << "CublasLt FP16 kernel average time: " << cublaslt_fp16_time << " ms" << std::endl;


    cudaMemcpy(h_C_cublaslt_fp16, d_C_fp16, half_size_C, cudaMemcpyDeviceToHost);
    for (int i=0; i<M*N; ++i) h_C_cpu[i] = __half2float(h_C_cublaslt_fp16[i]);

    bool cublas_fp16_correct = verifyResults(h_C_naive, h_C_cpu, 1e-2, size_C);
    std::cout << "cuBLAS FP32 results " << (cublas_fp16_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 1e-2." << std::endl;

    free(h_C_cublaslt_fp16);
    cudaFree(d_A_fp16);cudaFree(d_B_fp16);cudaFree(d_C_fp16);


    // Free memory
    free(h_A); free(h_B); free(h_C_naive); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;


}