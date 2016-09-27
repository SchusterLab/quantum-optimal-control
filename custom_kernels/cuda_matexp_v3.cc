

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include "dev_array.h"

using namespace tensorflow;
using namespace std;

REGISTER_OP("MatrixExp")
    .Attr("size: int")
    .Attr("input_num: int")
    .Attr("exp_num: int")
    .Input("coeff: float")
    .Input("matrix: float")
    .Output("output: float")
    .Doc(R"doc(
GPU kernel for exponentiating a matrix, that takes arbitrary input controls, and matrix_list as tensorflow constant.
)doc");

void matrixMultiplication(const float* A, const float* B, float* C, const int N);
void matrixAdd(const float* A, const float* B, const float coeff, float* C, const int N);
void matrixAddV2(const float* A, const float* B, const float* coeff, float* C, const int N);

class MatrixExpOp : public OpKernel {
 public:
  explicit MatrixExpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("size", &size_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("input_num", &input_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("exp_num", &exp_num_));
}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_0_tensor = context->input(0);
    auto input_0 = input_0_tensor.flat<float>();

    const Tensor& input_1_tensor = context->input(1);
    auto matrix_ = input_1_tensor.flat<float>();


    const int N = size_*size_;

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int>(sqrt(N)),static_cast<int>(sqrt(N))}),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();
 
    int dim = static_cast<int>(sqrt(N));
    
    dev_array<float> d_m_ii(N);

   
    dev_array<float> d_mat(N);
    dev_array<float> d_mat_temp(N);
    dev_array<float> d_mat_exp(N);
    dev_array<float> d_mat_exp_temp(N);
    dev_array<float> d_mat_n(N);
    dev_array<float> d_mat_n_temp(N);


    // Call the cuda kernel launcher

    d_mat.set(&matrix_.data()[0], N);

    for (int ii = 1; ii < input_num_; ii++) {
      d_m_ii.set(&matrix_.data()[ii*N], N);
      matrixAddV2(d_mat.getData(), d_m_ii.getData(), &input_0.data()[ii], d_mat_temp.getData(), dim);
      d_mat.set(d_mat_temp.getData(),N);
    }


    d_mat_n.set(&matrix_.data()[input_num_*N], N);
    d_mat_exp.set(&matrix_.data()[input_num_*N], N);

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
   int input_num_;
   int exp_num_;
};

REGISTER_KERNEL_BUILDER(Name("MatrixExp").Device(DEVICE_GPU), MatrixExpOp);

