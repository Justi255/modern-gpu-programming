#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 2048));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 10240));
  void* probs = ((void*)((char*)buf_dyn_shmem + 18432));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 20480));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 20992));
  float acc[8];
  float denom[2];
  float maximum[2];
  float scores[8];
  float updated[2];
  float correction[2];
  float rowsum[2];
  int length = Lengths[(((int)blockIdx.y) >> 1)];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(denom + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(maximum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
  uint4 condval;
  if ((((int)threadIdx.x) < 32)) {
    condval = *(uint4*)(Q + ((((int)blockIdx.y) * 256) + (((int)threadIdx.x) * 8)));
  } else {
    condval = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
  }
  *(uint4*)(((half_t*)Qs) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  for (int block = 0; block < ((max((min(length, ((((int)blockIdx.z) * 256) + 256)) - (((int)blockIdx.z) * 256)), 0) + 63) >> 6); ++block) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      if ((((((((int)blockIdx.z) * 256) + (block * 64)) + (i_1 * 16)) + (((int)threadIdx.x) >> 3)) < length) && (block < 4)) {
        *(uint4*)(((half_t*)Ks) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((((((int)blockIdx.y) >> 1) * 131072) + (((int)blockIdx.z) * 32768)) + (block * 8192)) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
        *(uint4*)(((half_t*)Vs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((((((int)blockIdx.y) >> 1) * 131072) + (((int)blockIdx.z) * 32768)) + (block * 8192)) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) & 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      } else {
        half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Ks) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
        half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Vs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_2 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    {
      half_t A_local[8];
      half_t B_local[8];
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[(((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float condval_1;
      if (((((((((((int)blockIdx.z) * 256) + (block * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_3 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_3 & 1)) < length) && (block < 4))) {
        condval_1 = (scores[i_3] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval_1 = -CUDART_INF_F;
      }
      scores[i_3] = condval_1;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      updated[i_4] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        updated[i_4] = max(updated[i_4], scores[((((rv & 1) * 4) + (i_4 * 2)) + (rv >> 1))]);
      }
      __syncthreads();
      updated[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(updated[i_4], (&(((float*)workspace_1)[0])));
      updated[i_4] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(updated[i_4]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      updated[i_5] = max(maximum[i_5], updated[i_5]);
      float condval_2;
      if ((updated[i_5] == -CUDART_INF_F)) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = updated[i_5];
      }
      correction[i_5] = exp2f((maximum[i_5] - condval_2));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_6 * 2));
        float condval_3;
        if ((updated[(i_6 & 1)] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[(i_6 & 1)];
        }
        float2 v__1 = make_float2(condval_3, condval_3);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_6 * 2)) = __1;
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      rowsum[i_7] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        rowsum[i_7] = (rowsum[i_7] + scores[((((rv_1 & 1) * 4) + (i_7 * 2)) + (rv_1 >> 1))]);
      }
      __syncthreads();
      rowsum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(rowsum[i_7], (&(((float*)workspace)[0])));
      rowsum[i_7] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(rowsum[i_7]);
    }
    float2 __3;
      float2 v__2 = *(float2*)(denom + 0);
      float2 v__3 = *(float2*)(correction + 0);
      float2 v__4 = *(float2*)(rowsum + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
    *(float2*)(denom + 0) = __3;
    *(float2*)(maximum + 0) = *(float2*)(updated + 0);
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_8 * 2));
        float2 v__6 = make_float2(correction[(i_8 & 1)], correction[(i_8 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_8 * 2)) = __4;
    }
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[(((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)scores[0]), ((half_t)scores[1])), __pack_half2(((half_t)scores[2]), ((half_t)scores[3])), __pack_half2(((half_t)scores[4]), ((half_t)scores[5])), __pack_half2(((half_t)scores[6]), ((half_t)scores[7])));
    {
      half_t A_local_1[8];
      half_t B_local_1[8];
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[(((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((ki_1 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    if ((((i_9 & 1) * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
      *(float2*)(Partial + ((((((((i_9 & 1) * 2048) + (((int)blockIdx.y) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + (((int)blockIdx.z) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_9 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_9 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      if (((i_10 * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        Maxima[((((i_10 * 32) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + ((int)blockIdx.z))] = maximum[i_10];
        Sums[((((i_10 * 32) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + ((int)blockIdx.z))] = denom[i_10];
      }
    }
  }
}

