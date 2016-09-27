

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include "dev_array.h"

using namespace tensorflow;
using namespace std;

REGISTER_OP("MatmulVecs")
    .Attr("size: int")
    .Attr("input_num: int")
    .Attr("vecs_num: int")
    .Attr("id: int")
    .Input("vecs: float")
    .Input("matrix: float")
    .Output("output: float")
    .Doc(R"doc(
GPU kernel for matrix vector multiplication. Used in backward propagation gradient.
)doc");

void matrixMultiplication(const float* A, const float* B, float* C, const int N, const int M);

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
    auto output = output_tensor->template flat<float>();
 
    int dim = static_cast<int>(sqrt(N));
    const int dim_m = vecs_num_;
    
   
    dev_array<float> d_mat(N);

    // Call the cuda kernel launcher

    d_mat.set(&matrix_.data()[id_*N], N);
    matrixMultiplication(d_mat.getData(), vecs.data(), output.data(), dim, dim_m);

    cudaDeviceSynchronize();     

    d_mat.free();

  }

    private:
   int size_;
   int input_num_;
   int vecs_num_;
   int id_;
};

REGISTER_KERNEL_BUILDER(Name("MatmulVecs").Device(DEVICE_GPU), MatmulVecsOp);

