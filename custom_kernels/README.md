## Example commands to build CUDA kernels:


cuda_matexp.so: GPU kernel for exponentiating a matrix
matrix_exp.so: CPU kernel for exponentiating a matrix

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

### building GPU CUDA kernel:

nvcc -std=c++11 -c -o matadd_coeff.cu.o matadd_coeff.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matadd_prepare.cu.o matadd_prepare.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matmul.cu.o matmul.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_matexp.so cuda_matexp.cc matadd_coeff.cu.o matmul.cu.o matadd_prepare.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart


### building CPU kernel:

g++ -std=c++11 -shared matrix_exp.cc -o matrix_exp.so -fPIC -I $TF_INC
