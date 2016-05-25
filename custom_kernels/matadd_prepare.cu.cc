#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void matrixPrepareKernel(const float* A,const float* B, const float* C, const float* coeff_a, const float* coeff_b, const float* coeff_c, float* D,const int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
         D[ROW * N + COL]= coeff_a[0] * A[ROW * N + COL] + coeff_b[0] * B[ROW * N + COL] + coeff_c[0] * C[ROW * N + COL];
    } 
}

void matrixPrepare(const float* A,const float* B, const float* C, const float* coeff_a, const float* coeff_b, const float* coeff_c, float* D,const int N){
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 1024){
            threadsPerBlock.x = 32;
            threadsPerBlock.y = 32;
            blocksPerGrid.x = static_cast<int>(ceil(double(N)/double(threadsPerBlock.x)));
            blocksPerGrid.y = static_cast<int>(ceil(double(N)/double(threadsPerBlock.y)));
        }


    matrixPrepareKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, coeff_a, coeff_b, coeff_c, D, N);

}


#endif
