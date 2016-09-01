#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include <vector>

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
    .Output("output: float32")
    .Doc(R"doc(
CPU kernel for exponentiating a matrix.
)doc");

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

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

  std::vector<float> matmul(std::vector<float> input_mat_1, std::vector<float> input_mat_2){
    const int N = input_mat_1.size();
    float elem;
    std::vector<float> output_mat;    
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
      output_mat.push_back(elem);
    }
    return output_mat;
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

    std::vector<float> mat(N);

    for (int i = 0; i < N; i++) {
      mat[i] = input_0(0) * matrix_0_[i] + input_1(0) * matrix_1_[i] + input_2(0) * matrix_2_[i];
    }

    std::vector<float> mat_exp = matrix_I_;
    std::vector<float> mat_n = mat; 
    
    for (int i = 0; i < N; i++) {
        mat_exp[i] += mat[i];
    }

    float factorial = 1.0;    

    for (int num  = 2; num < exp_num_+1; num++) {
      factorial = factorial * num;
      mat_n = matmul(mat_n,mat);
      for (int i = 0; i < N; i++) {
        mat_exp[i] += mat_n[i]/factorial;
      }
    }


    auto output = output_tensor->template flat<float>();

    for (int i = 0; i < N; i++) {
      output(i) = mat_exp[i];
    }

  }
  private:
   int size_;
   int exp_num_;
   std::vector<float> matrix_0_;
   std::vector<float> matrix_1_;
   std::vector<float> matrix_2_;
   std::vector<float> matrix_I_;
};

REGISTER_KERNEL_BUILDER(Name("MatrixExp").Device(DEVICE_CPU), MatrixExpOp);
