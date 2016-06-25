#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void matrixAddV4Kernel(const float* A,const float* B, const float* coeff, const float coeff_2, float* C,const int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
         C[ROW * N + COL]= A[ROW * N + COL] + coeff_2 * coeff[0] * B[ROW * N + COL];
    } 
}

void matrixAddV4(const float* A, const float* B, const float* coeff,const float coeff_2, float* C, const int N){
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 1024){
            threadsPerBlock.x = 32;
            threadsPerBlock.y = 32;
            blocksPerGrid.x = static_cast<int>(ceil(double(N)/double(threadsPerBlock.x)));
            blocksPerGrid.y = static_cast<int>(ceil(double(N)/double(threadsPerBlock.y)));
        }

    matrixAddV4Kernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, coeff, coeff_2, C, N);

}


#endif
