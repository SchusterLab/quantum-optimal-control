#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void matrixMultiplicationKernel(const float* A,const float* B, float* C,const int N, const int M) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    
    if (ROW < N && COL < M) {
        float tmpSum = 0;
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * M + COL];
        }
        C[ROW * M + COL] = tmpSum;
    }

}


void matrixMultiplication(const float *A, const float *B, float *C, const int N, const int M){
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 1024){
            threadsPerBlock.x = 32;
            threadsPerBlock.y = 32;
            blocksPerGrid.x = static_cast<int>(ceil(double(N)/double(threadsPerBlock.x)));
            blocksPerGrid.y = static_cast<int>(ceil(double(N)/double(threadsPerBlock.y)));
        }


    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N, M);

}


#endif
