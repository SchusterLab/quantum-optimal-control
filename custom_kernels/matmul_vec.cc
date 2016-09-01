#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include <vector>

REGISTER_OP("MatmulVecs")
    .Attr("size: int")
    .Attr("input_num: int")
    .Attr("vecs_num: int")
    .Attr("id: int")
    .Input("vecs: float")
    .Input("matrix: float")
    .Output("output: float")
    .Doc(R"doc(
CPU kernel for matrix vector multiplication. Used in backward propagation gradient.
)doc");

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class MatmulVecsOp : public OpKernel {
 public:
  explicit MatmulVecsOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context,
                   context->GetAttr("size", &size_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("input_num", &input_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("vecs_num", &vecs_num_));
  OP_REQUIRES_OK(context,
                   context->GetAttr("id", &id_));
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
    const Tensor& input_1_tensor = context->input(0);
    auto vecs = input_1_tensor.flat<float>();

    const Tensor& input_2_tensor = context->input(1);
    auto matrix_ = input_2_tensor.flat<float>();

    const int N = size_*size_;

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int>(sqrt(N)),vecs_num_}),
                                                     &output_tensor));

    std::vector<float> mat(N);

    mat = matset(&matrix_.data()[0],id_,N); 

    std::vector<float> mat_n = matset(&vecs.data()[0],0,size_*vecs_num_);


    mat_n = mat_vec_mul(mat,mat_n,size_,vecs_num_);


    auto output = output_tensor->template flat<float>();

    for (int i = 0; i < size_*vecs_num_; i++) {
      output(i) = mat_n[i];
    }

  }
  private:
   int size_;
   int input_num_;
   int vecs_num_; 
   int id_; 
};

REGISTER_KERNEL_BUILDER(Name("MatmulVecs").Device(DEVICE_CPU), MatmulVecsOp);
