export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

\rm -r ./build

mkdir build
cd build

nvcc -std=c++11 -c -o matadd_coeff.cu.o ../matadd_coeff.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matadd_prepare.cu.o ../matadd_prepare.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matmul.cu.o ../matmul.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matadd_coeff_v2.cu.o ../matadd_coeff_v2.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matadd_coeff_v3.cu.o ../matadd_coeff_v3.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o matadd_coeff_v4.cu.o ../matadd_coeff_v4.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o matrix_exp.so ../matrix_exp.cc -fPIC -I $TF_INC

g++ -std=c++11 -shared -o cuda_matexp.so ../cuda_matexp.cc matadd_coeff.cu.o matmul.cu.o matadd_prepare.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -L /usr/local/cuda/lib64 -lcudart

g++ -std=c++11 -shared -o cuda_matexp_v2.so ../cuda_matexp_v2.cc matadd_coeff.cu.o matmul.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -L /usr/local/cuda/lib64 -lcudart

g++ -std=c++11 -shared -o cuda_matexp_v3.so ../cuda_matexp_v3.cc matadd_coeff.cu.o matmul.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

g++ -std=c++11 -shared -o cuda_matexp_v4.so ../cuda_matexp_v4.cc matadd_coeff.cu.o matmul.cu.o matadd_coeff_v2.cu.o matadd_coeff_v4.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

nvcc -std=c++11 -c -o matmul_v2.cu.o ../matmul_v2.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_matexp_vecs.so ../cuda_matexp_vecs.cc matadd_coeff_v3.cu.o matmul_v2.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

g++ -std=c++11 -shared -o cuda_matexp_vecs_grads.so ../cuda_matexp_vecs_grads.cc matadd_coeff_v3.cu.o matmul_v2.cu.o matadd_coeff_v2.cu.o -I $TF_INC -I /usr/local/cuda/include -fPIC -lcudart

