#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include <vector>
#include <cmath>

REGISTER_OP("MatrixExp")
    .Attr("size: int")
    .Attr("input_num: int")
    .Attr("exp_num: int")
    .Attr("div: int")
    .Input("coeff: float")
    .Input("matrix: float")
    .Output("output: float")
    .Doc(R"doc(
CPU kernel for exponentiating a matrix, that takes arbitrary input controls, and matrix_list as tensorflow constant, and with scaling and squaring capability.
)doc");

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class MatrixExpOp : public OpKernel {
 public:
  explicit MatrixExpOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context,
                   context->GetAttr("size", &size_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("input_num", &input_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("exp_num", &exp_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("div", &div_));
  }

  std::vector<float> matmul(std::vector<float> input_mat_1, std::vector<float> input_mat_2, int N){
    float elem;
    std::vector<float> output_mat(N);    
    const int d = static_cast<int>(sqrt(N));
    int i;
    int j;

    for (int n = 0; n < N ; n++) {

      i = n / d;
      j = n % d;
      elem = 0;
      for (int k = 0; k < d ; k++) {
        elem += input_mat_1[i*d + k] * input_mat_2[k*d + j];
      }
      output_mat[n] = elem;
    }
    return output_mat;
  }

  std::vector<float> matadd(std::vector<float> input_mat_1, std::vector<float> input_mat_2, int N){
    std::vector<float> output_mat(N);

    for (int n = 0; n < N ; n++) {
      output_mat[n] = input_mat_1[n]+input_mat_2[n];
    }
    return output_mat;
  }

  std::vector<float> matscale(const float scale_1, const float* scale_2, std::vector<float> input_mat, int N){
    std::vector<float> output_mat(N);

    for (int n = 0; n < N ; n++) {
      output_mat[n] = scale_1 * scale_2[0] * input_mat[n];
    }
    return output_mat;
  }

  std::vector<float> matset(const float* input_mat,int id, int N){
    std::vector<float> output_mat(N);

    for (int n = 0; n < N ; n++) {
      output_mat[n] = input_mat[id*N + n];
    }
    return output_mat;
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

    std::vector<float> mat(N);

    float div_factor = pow(2.0, div_);

    mat = matscale(1./div_factor,&input_0.data()[0],matset(&matrix_.data()[0],0,N),N);

    for (int ii = 1; ii < input_num_; ii++) {
      mat = matadd(mat,matscale(1./div_factor,&input_0.data()[ii],matset(&matrix_.data()[0],ii,N),N),N);
    }
    

    std::vector<float> mat_exp = matset(&matrix_.data()[0],input_num_,N);
    std::vector<float> mat_n = mat; 
    
    for (int i = 0; i < N; i++) {
        mat_exp[i] += mat[i];
    }

    float factorial = 1.0;    

    for (int num  = 2; num < exp_num_+1; num++) {
      factorial = factorial * num;
      mat_n = matmul(mat_n,mat,N);
      for (int i = 0; i < N; i++) {
        mat_exp[i] += mat_n[i]/factorial;
      }
    }

    for (int num = 0; num < div_; num++) {
      mat_exp = matmul(mat_exp,mat_exp,N);
    }


    auto output = output_tensor->template flat<float>();

    for (int i = 0; i < N; i++) {
      output(i) = mat_exp[i];
    }

  }
  private:
   int size_;
   int input_num_;
   int exp_num_;
   int div_;
};

REGISTER_KERNEL_BUILDER(Name("MatrixExp").Device(DEVICE_CPU), MatrixExpOp);
