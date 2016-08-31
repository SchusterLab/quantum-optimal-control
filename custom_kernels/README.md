## Example commands to build CUDA kernels:

**dev_array.h**: a helper function for allocate and release of GPU memory

**matadd_coeff.cu.o**: C=A+coeff*B, coeff as float

**matadd_coeff_v2.cu.o**:C=A+coeff*B, coeff as float pointer

**matadd_coeff_v3.cu.o**: C=A+coeff*B, coeff as float, with N*M dimension (for matrix-vec)

**matadd_coeff_v4.cu.o**: C=A+coeff*coeff_2*B, coeff as float pointer, coeff_2 as float

**matmul.cu.o**: C=A*B

**matmul_v2.cu.o**: C=A*B, with N*M dimension (for matrix-vec)

**cuda_matexp.so**: GPU kernel for exponentiating a matrix

**cuda_matexp_v2.so**: GPU kernel for exponentiating a matrix, that takes arbitrary input controls

**cuda_matexp_v3.so**: GPU kernel for exponentiating a matrix, that takes arbitrary input controls, and matrix_list as tensorflow constant

**cuda_matexp_v4.so**: GPU kernel for exponentiating a matrix, that takes arbitrary input controls, and matrix_list as tensorflow constant, and with scaling and squaring capability

**matrix_exp.so**: CPU kernel for exponentiating a matrix

cuda_matexp_vecs.so: GPU kernel for exp matrix - vector multiplication

cuda_matexp_vecs_grads.so: Identitcal as cuda_matexp_vecs.so, but with different name such that graident python file wont be confused, as this operation is required in gradient calculation.

cuda_matexp_vecs_v2.so: GPU kernel for exp matrix - vector multiplication, that takes arbitrary input controls, and matrix_list as tensorflow constant

cuda_matexp_vecs_grads_v2.so: Identitcal as cuda_matexp_vecs_v2.so, but with different name such that graident python file wont be confused, as this operation is required in gradient calculation.

cuda_matmul_vec.cc: For matrix vector multiplication. Used in backward propagation gradient.
