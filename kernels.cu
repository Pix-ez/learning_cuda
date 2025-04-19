
#include <cuda_runtime.h>


__global__ void naiveMatrixMultiply(float *A, float *B, float *C, int M, int N, int K){
    int row = blockIdx.y* blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum =0.0f;
        for (int i=0 ; i<K; ++i){
            sum += A[row*K +i] * B[i*N +col];
        }
        C[row*N + col] = sum;
    }
}

