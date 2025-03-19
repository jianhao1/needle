#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t convert_index(size_t index, CudaVec shape, CudaVec strides, size_t offset)
{
  size_t res = offset;
  for (int i = shape.size - 1; i >= 0; i--)
  {
    int j = index % shape.data[i];
    index /= shape.data[i];
    res += strides.data[i] * j;
  }
  return res;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size)
    out[gid] = a[convert_index(gid, shape, strides, offset)];
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset)
{
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size)
    out[convert_index(gid, shape, strides, offset)] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset)
{
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size)
    out[convert_index(gid, shape, strides, offset)] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

#define DEFINE_EWISE_OP_KERNEL(Op) \
  __global__ void Ewise##Op##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid < size) out[gid] = Op(a[gid], b[gid]); \
  }

#define DEFINE_EWISE_OP_FUNC(Op) \
  void Ewise##Op(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
    CudaDims dim = CudaOneDim(out->size); \
    Ewise##Op##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
  }

#define DEFINE_EWISE_UNARY_OP_KERNEL(Op) \
  __global__ void Ewise##Op##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid < size) out[gid] = Op(a[gid]); \
  }

#define DEFINE_EWISE_UNARY_OP_FUNC(Op) \
  void Ewise##Op(const CudaArray& a, CudaArray* out) { \
    CudaDims dim = CudaOneDim(out->size); \
    Ewise##Op##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
  }

#define DEFINE_SCALAR_OP_KERNEL(Op) \
  __global__ void Scalar##Op##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid < size) out[gid] = Op(a[gid], val); \
  }

#define DEFINE_SCALAR_OP_FUNC(Op) \
  void Scalar##Op(const CudaArray& a, scalar_t val, CudaArray* out) { \
    CudaDims dim = CudaOneDim(out->size); \
    Scalar##Op##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
  }

__device__ scalar_t Mul(scalar_t a, scalar_t b) {
  return a * b;
}

__device__ scalar_t Div(scalar_t a, scalar_t b) {
  return a / b;
}

__device__ scalar_t Power(scalar_t a, scalar_t b) {
  return pow(a, b);
}

__device__ scalar_t Maximum(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

__device__ scalar_t Eq(scalar_t a, scalar_t b) {
  return a == b;
}

__device__ scalar_t Ge(scalar_t a, scalar_t b) {
  return a >= b;
}

__device__ scalar_t Log(scalar_t a) {
  return log(a);
}

__device__ scalar_t Exp(scalar_t a) {
  return exp(a);
}

__device__ scalar_t Tanh(scalar_t a) {
  return tanh(a);
}

DEFINE_EWISE_OP_KERNEL(Mul)
DEFINE_EWISE_OP_FUNC(Mul)
DEFINE_SCALAR_OP_KERNEL(Mul)
DEFINE_SCALAR_OP_FUNC(Mul)
DEFINE_EWISE_OP_KERNEL(Div)
DEFINE_EWISE_OP_FUNC(Div)
DEFINE_SCALAR_OP_KERNEL(Div)
DEFINE_SCALAR_OP_FUNC(Div)
DEFINE_SCALAR_OP_KERNEL(Power)
DEFINE_SCALAR_OP_FUNC(Power)
DEFINE_EWISE_OP_KERNEL(Maximum)
DEFINE_EWISE_OP_FUNC(Maximum)
DEFINE_SCALAR_OP_KERNEL(Maximum)
DEFINE_SCALAR_OP_FUNC(Maximum)
DEFINE_EWISE_OP_KERNEL(Eq)
DEFINE_EWISE_OP_FUNC(Eq)
DEFINE_SCALAR_OP_KERNEL(Eq)
DEFINE_SCALAR_OP_FUNC(Eq)
DEFINE_EWISE_OP_KERNEL(Ge)
DEFINE_EWISE_OP_FUNC(Ge)
DEFINE_SCALAR_OP_KERNEL(Ge)
DEFINE_SCALAR_OP_FUNC(Ge)
DEFINE_EWISE_UNARY_OP_KERNEL(Log)
DEFINE_EWISE_UNARY_OP_FUNC(Log)
DEFINE_EWISE_UNARY_OP_KERNEL(Exp)
DEFINE_EWISE_UNARY_OP_FUNC(Exp)
DEFINE_EWISE_UNARY_OP_KERNEL(Tanh)
DEFINE_EWISE_UNARY_OP_FUNC(Tanh)

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication
////////////////////////////////////////////////////////////////////////////////

#define L 16
#define S 8
#define V 8

__device__ void cooperative_fetch_slice(const scalar_t *a, int a_m, int a_n, int offset_y, int offset_x, scalar_t *s, int s_m, int s_n)
{
  int nthreads = blockDim.y * blockDim.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < s_m * s_n; i += nthreads)
  {
    int j = i + tid;
    if (j >= s_m * s_n)
      break;
    int y = j / s_n;
    int x = j % s_n;
    if (y + offset_y >= a_m || x + offset_x >= a_n)
      s[y*s_n+x] = 0;
    else
      s[y*s_n+x] = a[convert_index(j, {2, {s_m, s_n}}, {2, {a_n, 1}}, offset_y * a_n + offset_x)];
  }
}

__global__ void MatmulKernel(const scalar_t *A, const scalar_t *B, scalar_t *out, uint32_t M, uint32_t N,
            uint32_t P)
{
  __shared__ scalar_t sA[L][S], sB[S][L];
  scalar_t c[V][V] = {0};
  scalar_t a[V], b[V];

  for (int ko = 0; ko < N; ko += S)
  {
    /* Cooperative Fetching */
    // sA
    cooperative_fetch_slice(A, M, N, blockIdx.y * L, ko, sA[0], L, S);
    // sB
    cooperative_fetch_slice(B, N, P, ko, blockIdx.x * L, sB[0], S, L);
    __syncthreads();

    for (int ki = 0; ki < S; ki++)
    {
      for (int i = 0; i < V; i++)
      {
        a[i] = sA[threadIdx.y * V + i][ki];
        b[i] = sB[ki][threadIdx.x * V + i];
      }
      for (int y = 0; y < V; y++)
      {
        for (int x = 0; x < V; x++)
        {
          c[y][x] += a[y] * b[x];
        }
      }
    }
  }

  int ybase = (blockIdx.y * blockDim.y + threadIdx.y) * V;
  int xbase = (blockIdx.x * blockDim.x + threadIdx.x) * V;
  for (int y = 0; y < V; y++)
  {
    for (int x = 0; x < V; x++)
    {
      if (ybase + y >= M || xbase + x >= P)
      {
        continue;
      }
      out[(ybase+y)*P+xbase+x] = c[y][x];
    }
  }

}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  CudaDims dim;
  dim.block = dim3(L/V, L/V);
  dim.grid = dim3((P+L-1)/L, (M+L-1)/L);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
  {
    int offset = gid * reduce_size;
    out[gid] = a[offset];
    for (int i = offset + 1; i < offset + reduce_size; i++)
    {
      if (a[i] > out[gid])
      {
        out[gid] = a[i];
      }
    }
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
  {
    int offset = gid * reduce_size;
    out[gid] = 0;
    for (int i = offset; i < offset + reduce_size; i++)
    {
      out[gid] += a[i];
    }
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
