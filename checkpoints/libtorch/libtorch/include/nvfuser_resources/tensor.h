// Generated from "/pytorch/torch/csrc/jit/codegen/cuda/runtime/tensor.cu"
// 2021-10-15 04:33:19

namespace nvfuser_resources {

constexpr const char* tensor_cu = R"(
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int int16_t;
typedef long long int int64_t;

template <typename T, int N>
struct Tensor {
  __device__ T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  int64_t size[N];
  int64_t stride[N];
};

// Specialization for 0-dim case as it does not need size and stride arrays.
// They will be an error as well since zero-length arrays are not allowed.
template <typename T>
struct Tensor<T, 0> {
  __device__ T& operator[](int64_t) {
    return *data;
  };

  T* data;
};
)";

} // namespace nvfuser_resources
