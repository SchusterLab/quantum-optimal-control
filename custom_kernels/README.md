## Example commands to build CUDA kernels:

dev_array.h: a helper function for allocate and release of GPU memory

cuda_matexp.so: GPU kernel for exponentiating a matrix

cuda_matexp_v2.so: GPU kernel for exponentiating a matrix, that takes arbitrary input controls

matrix_exp.so: CPU kernel for exponentiating a matrix

cuda_matexp_vecs.so: GPU kernel for exp matrix - vector multiplication

#### Run these commands in the folder ./build

### First run

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

### building GPU CUDA kernel:

nvcc -std=c++11 -c -o matadd_coeff.cu.o ../matadd_coeff.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matadd_prepare.cu.o ../matadd_prepare.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matmul.cu.o ../matmul.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_matexp.so ../cuda_matexp.cc matadd_coeff.cu.o matmul.cu.o matadd_prepare.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

nvcc -std=c++11 -c -o matadd_coeff_v2.cu.o ../matadd_coeff_v2.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_matexp_v2.so ../cuda_matexp_v2.cc matadd_coeff.cu.o matmul.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

g++ -std=c++11 -shared -o cuda_matexp_v3.so ../cuda_matexp_v3.cc matadd_coeff.cu.o matmul.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

g++ -std=c++11 -shared -o cuda_matexp_v3.so ../cuda_matexp_v3.cc matadd_coeff.cu.o matmul.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

nvcc -std=c++11 -c -o matadd_coeff_v3.cu.o ../matadd_coeff_v3.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matmul_v2.cu.o ../matmul_v2.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_matexp_vecs.so ../cuda_matexp_vecs.cc matadd_coeff_v3.cu.o matmul_v2.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

g++ -std=c++11 -shared -o cuda_matexp_vecs_grads.so ../cuda_matexp_vecs_grads.cc matadd_coeff_v3.cu.o matmul_v2.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

### building CPU kernel:

g++ -std=c++11 -shared ../matrix_exp.cc -o matrix_exp.so -fPIC -I $TF_INC
