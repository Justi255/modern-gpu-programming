#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const float* __restrict__ Maxima, half_t* __restrict__ Out, const float* __restrict__ Partial, const float* __restrict__ Sums);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const float* __restrict__ Maxima, half_t* __restrict__ Out, const float* __restrict__ Partial, const float* __restrict__ Sums) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 0));
  float maximum[1];
  float global_max[1];
  float weights[1];
  float normalizer[1];
  float values[2];
  float result[2];
  maximum[0] = Maxima[(((((int)blockIdx.y) * 32) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) >> 5))];
  global_max[0] = -CUDART_INF_F;
  global_max[0] = max(global_max[0], maximum[0]);
  global_max[0] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(global_max[0], (&(((float*)workspace_2)[0])));
  float condval;
  if ((global_max[0] == -CUDART_INF_F)) {
    condval = 0x0p+0f/*0.000000e+00*/;
  } else {
    condval = global_max[0];
  }
  maximum[0] = exp2f((maximum[0] - condval));
  weights[0] = (maximum[0] * Sums[(((((int)blockIdx.y) * 32) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) >> 5))]);
  normalizer[0] = 0x0p+0f/*0.000000e+00*/;
  normalizer[0] = (normalizer[0] + weights[0]);
  __syncthreads();
  normalizer[0] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(normalizer[0], (&(((float*)workspace_1)[0])));
  float2 __1;
    float2 v_ = make_float2(maximum[0], maximum[0]);
    float2 v__1 = *(float2*)(Partial + (((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) * 2)));
    *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
  *(float2*)(values + 0) = __1;
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    result[i] = 0x0p+0f/*0.000000e+00*/;
    result[i] = (result[i] + values[i]);
    __syncthreads();
    result[i] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(result[i], (&(((float*)workspace)[0])));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    {
      half_t Out_local_cast[2];
      uint1 __2;
      float2 __3;
        float2 v__2 = *(float2*)(result + 0);
        float2 v__3 = make_float2(max(normalizer[0], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/), max(normalizer[0], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/));
        __3.x = (v__2.x/v__3.x);
        __3.y = (v__2.y/v__3.y);
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&__3))[0]);
      *(uint1*)(Out_local_cast + 0) = __2;
      *(uint1*)(Out + (((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 31) * 2))) = *(uint1*)(Out_local_cast + 0);
    }
  }
}

