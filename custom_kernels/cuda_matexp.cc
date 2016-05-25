/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include "dev_array.h"

using namespace tensorflow;
using namespace std;

REGISTER_OP("MatrixExp")
    .Attr("size: int")
    .Attr("exp_num: int")
    .Attr("matrix_0: list(float)")
    .Attr("matrix_1: list(float)")
    .Attr("matrix_2: list(float)")
    .Attr("matrix_I: list(float)")
    .Input("coeff_0: float32")
    .Input("coeff_1: float32")
    .Input("coeff_2: float32")
    .Output("output: float")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.

output: A Tensor.
  output = input + 1
)doc");

void matrixMultiplication(const float* A, const float* B, float* C, const int N);
void matrixAdd(const float* A, const float* B, const float coeff, float* C, const int N);
void matrixPrepare(const float* A,const float* B, const float* C, const float* coeff_a, const float* coeff_b, const float* coeff_c, float* D,const int N);

class MatrixExpOp : public OpKernel {
 public:
  explicit MatrixExpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("size", &size_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("exp_num", &exp_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("matrix_0", &matrix_0_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("matrix_1", &matrix_1_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("matrix_2", &matrix_2_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("matrix_I", &matrix_I_));
}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_0_tensor = context->input(0);
    auto input_0 = input_0_tensor.flat<float>();

    const Tensor& input_1_tensor = context->input(1);
    auto input_1 = input_1_tensor.flat<float>();

    const Tensor& input_2_tensor = context->input(2);
    auto input_2 = input_2_tensor.flat<float>();

    const int N = matrix_0_.size();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int>(sqrt(N)),static_cast<int>(sqrt(N))}),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
 
    int dim = static_cast<int>(sqrt(N));
    
    dev_array<float> d_m0(N);
    dev_array<float> d_m1(N);
    dev_array<float> d_m2(N);

    d_m0.set(&matrix_0_[0], N);
    d_m1.set(&matrix_1_[0], N);
    d_m2.set(&matrix_2_[0], N);
   
    dev_array<float> d_mat(N);
    dev_array<float> d_mat_exp(N);
    dev_array<float> d_mat_exp_temp(N);
    dev_array<float> d_mat_n(N);
    dev_array<float> d_mat_n_temp(N);


    // Call the cuda kernel launcher
    matrixPrepare(d_m0.getData(), d_m1.getData(), d_m2.getData(),input_0.data(), input_1.data(),input_2.data(), d_mat.getData(), dim);

    d_mat_n.set(&matrix_I_[0], N);
    d_mat_exp.set(&matrix_I_[0], N);

    float inv_factorial = 1.0;
    
    matrixMultiplication(d_mat_n.getData(), d_mat.getData(), d_mat_n_temp.getData(), dim);
    matrixAdd(d_mat_exp.getData(), d_mat_n_temp.getData(), inv_factorial, d_mat_exp_temp.getData(), dim);
    
    d_mat_n.set(d_mat_n_temp.getData(), N);
    d_mat_exp.set(d_mat_exp_temp.getData(), N);


    for (int num  = 2; num < exp_num_; num++) {
      inv_factorial = inv_factorial/ num;
      matrixMultiplication(d_mat_n.getData(), d_mat.getData(), d_mat_n_temp.getData(), dim);
      matrixAdd(d_mat_exp.getData(), d_mat_n_temp.getData(), inv_factorial, d_mat_exp_temp.getData(), dim);
    
      d_mat_n.set(d_mat_n_temp.getData(), N);
      d_mat_exp.set(d_mat_exp_temp.getData(), N);
    }
    
    inv_factorial = inv_factorial/ exp_num_;
    matrixMultiplication(d_mat_n.getData(), d_mat.getData(), d_mat_n_temp.getData(), dim);
    matrixAdd(d_mat_exp.getData(), d_mat_n_temp.getData(), inv_factorial, output.data(), dim);
    cudaDeviceSynchronize();     


  }

    private:
   int size_;
   int exp_num_;
   std::vector<float> matrix_0_;
   std::vector<float> matrix_1_;
   std::vector<float> matrix_2_;
   std::vector<float> matrix_I_;
};

REGISTER_KERNEL_BUILDER(Name("MatrixExp").Device(DEVICE_GPU), MatrixExpOp);

