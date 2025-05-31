#include <cuda_runtime.h>
#include "config.h"

__global__ void naiveMatrixMultiply(float *A, float *B, float *C, int M, int N, int K){
    
    // no coalesced memory
    int row = blockIdx.y* blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    
    // no coalesced memory

    if (row < M && col < N) {
        float sum =0.0f;
        for (int i=0 ; i<K; ++i){
            sum += A[row*K +i] * B[i*N +col];
        }
        C[row*N + col] = sum;
    }
}


//GEMEM coalesed 
// template <const uint BLOCKSIZE>
//for now just hardcode 16 blocksize
#define BLOCKSIZE 16
__global__ void gememMatrixMultiply(float *A, float *B, float *C, int M, int N, int K){
    
    // no coalesced memory
    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    
    // no coalesced memory

    if (row < M && col < N) {
        float sum =0.0f;
        for (int i=0 ; i<K; ++i){
            sum += A[row*K +i] * B[i*N +col];
        }
        C[row*N + col] = sum;
    }
}
//Tiled kernel for fast memory access result in fast compute rather than waiting for access element from global mem.
__global__ void tiledMatrixMultiply(float *A, float *B, float *C, int M, int N, int K){
    //first load chunk of data into faster cache which is fast rather than going for every element to main memory called memory coalescing
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE]; 
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE]; 

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y; 

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

   

    float sum = 0.0f;

    //first we need to load tile chunk elements from global memory to per block's shared memory
    for (int tile =0; tile< (K +TILE_SIZE-1) / TILE_SIZE; ++tile){
        if (row < M && tile *TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K +tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;
        
        if(col <N && tile *TILE_SIZE +ty <K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N +col];
        else
            sharedB[ty][tx] = 0.0f;

    //Copying is done we can esure that using syncThreads()
    __syncthreads(); //it will wait for all thread to finish work 

    //now we do multiply operation
    for (int k=0; k < TILE_SIZE; ++k)
        sum += sharedA[ty][k] * sharedB[k][tx];

    //again we wait for all thread to finish work
    __syncthreads();
    //now copy answer to output matrix in global memory
    if (row<M && col < N)
        C[row *N +col] = sum;

    }
}