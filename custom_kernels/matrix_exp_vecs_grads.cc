#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include <vector>

REGISTER_OP("MatrixExpVecsGrads")
    .Attr("size: int")
    .Attr("input_num: int")
    .Attr("exp_num: int")
    .Attr("vecs_num: int")
    .Input("coeff: float")
    .Input("vecs: float")
    .Input("matrix: float")
    .Output("output: float")
    .Doc(R"doc(
Identitcal as cuda_matexp_vecs_v2.so, but with different name such that graident python file wont be confused, as this operation is required in gradient calculation.
)doc");

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class MatrixExpVecsGradsOp : public OpKernel {
 public:
  explicit MatrixExpVecsGradsOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context,
                   context->GetAttr("size", &size_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("input_num", &input_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("exp_num", &exp_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("vecs_num", &vecs_num_));
  }

  std::vector<float> mat_vec_mul(std::vector<float> input_mat_1, std::vector<float> input_mat_2, int d_N, int d_M){
    float elem;
    std::vector<float> output_mat(d_N*d_M);    
    int i;
    int j;

    for (int n = 0; n < d_N*d_M ; n++) {

      i = n / d_M;
      j = n % d_M;
      elem = 0;
      for (int k = 0; k < d_N ; k++) {
        elem += input_mat_1[i*d_N + k] * input_mat_2[k*d_M + j];
      }
      output_mat[n] = elem;
    }
    return output_mat;
  }

  std::vector<float> mataddscale(std::vector<float> input_mat_1, std::vector<float> input_mat_2, const float scale,int N){
    std::vector<float> output_mat(N);

    for (int n = 0; n < N ; n++) {
      output_mat[n] = input_mat_1[n]+scale*input_mat_2[n];
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
    auto vecs = input_1_tensor.flat<float>();

    const Tensor& input_2_tensor = context->input(2);
    auto matrix_ = input_2_tensor.flat<float>();

    const int N = size_*size_;

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int>(sqrt(N)),vecs_num_}),
                                                     &output_tensor));

    std::vector<float> mat(N);

    mat = matscale(1.,&input_0.data()[0],matset(&matrix_.data()[0],0,N),N);

    for (int ii = 1; ii < input_num_; ii++) {
      mat = matadd(mat,matscale(1.,&input_0.data()[ii],matset(&matrix_.data()[0],ii,N),N),N);
    }
    

    std::vector<float> mat_exp = matset(&vecs.data()[0],0,size_*vecs_num_);
    std::vector<float> mat_n = matset(&vecs.data()[0],0,size_*vecs_num_);

    float factorial = 1.0;    

    for (int num  = 1; num < exp_num_+1; num++) {
      factorial = factorial * num;
      mat_n = mat_vec_mul(mat,mat_n,size_,vecs_num_);
      mat_exp = mataddscale(mat_exp, mat_n,1./factorial,size_*vecs_num_);
    }


    auto output = output_tensor->template flat<float>();

    for (int i = 0; i < size_*vecs_num_; i++) {
      output(i) = mat_exp[i];
    }

  }
  private:
   int size_;
   int input_num_;
   int exp_num_;
   int vecs_num_;  
};

REGISTER_KERNEL_BUILDER(Name("MatrixExpVecsGrads").Device(DEVICE_CPU), MatrixExpVecsGradsOp);
