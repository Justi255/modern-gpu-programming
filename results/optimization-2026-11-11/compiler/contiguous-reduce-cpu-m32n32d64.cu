#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const int* __restrict__ Table, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const int* __restrict__ Table, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 4096));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 12288));
  void* probs = ((void*)((char*)buf_dyn_shmem + 20480));
  void* reduction_shared = ((void*)((char*)buf_dyn_shmem + 22528));
  float acc[16];
  float denom[2];
  float maximum[2];
  float scores[8];
  float segmented[8];
  float partial_warp[2];
  float updated[2];
  float correction[2];
  float rowsum[2];
  int length = Lengths[0];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(denom + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(maximum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(((half_t*)Qs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((i_1 * 1024) + (((int)threadIdx.x) * 8)));
  }
  if (0 < (min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512))) {
    if (((((int)blockIdx.z) * 512) + 32) <= length) {
      for (int pg = 0; pg < 2; ++pg) {
        int physical = Table[((((int)blockIdx.z) * 32) + pg)];
        tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((int64_t)physical) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
        tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((int64_t)physical) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
      }
      tl::cp_async_commit();
    } else {
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        if ((((((int)blockIdx.z) * 512) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < length) {
          int physical_1 = Table[((((int)blockIdx.z) * 32) + i_2)];
          *(uint4*)(((half_t*)Ks) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((int64_t)physical_1) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
          *(uint4*)(((half_t*)Vs) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((int64_t)physical_1) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        } else {
          half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Ks) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
          half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Vs) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
        }
      }
    }
  }
  for (int block = 0; block < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 31) >> 5); ++block) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((block + 1) < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 31) >> 5)) {
      if (((((((int)blockIdx.z) * 512) + (block * 32)) + 64) <= length) && (block <= 14)) {
        for (int pg_1 = 0; pg_1 < 2; ++pg_1) {
          int physical_2 = Table[((((((int)blockIdx.z) * 32) + (block * 2)) + pg_1) + 2)];
          tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((((block + 1) & 1) * 2048) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((int64_t)physical_2) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
          tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((((block + 1) & 1) * 2048) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((int64_t)physical_2) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          if (((((((((int)blockIdx.z) * 512) + (block * 32)) + (i_3 * 16)) + (((int)threadIdx.x) >> 3)) + 32) < length) && (block < 15)) {
            int physical_3 = Table[((((((int)blockIdx.z) * 32) + (block * 2)) + i_3) + 2)];
            *(uint4*)(((half_t*)Ks) + ((((((((block + 1) & 1) * 2048) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((int64_t)physical_3) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
            *(uint4*)(((half_t*)Vs) + ((((((((block + 1) & 1) * 2048) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((int64_t)physical_3) * (int64_t)1024) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
          } else {
            half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Ks) + ((((((((block + 1) & 1) * 2048) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
            half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Vs) + ((((((((block + 1) & 1) * 2048) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
          }
        }
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_4 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
    }
    {
      half_t A_local[8];
      half_t B_local[8];
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((block & 1) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      float condval;
      if (((((((((((int)blockIdx.z) * 512) + (block * 32)) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_5 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) < length) && (block < 16))) {
        condval = (scores[i_5] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval = -CUDART_INF_F;
      }
      scores[i_5] = condval;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      *(float2*)(segmented + (i_6 * 2)) = *(float2*)(scores + (((i_6 & 1) * 4) + ((i_6 >> 1) * 2)));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      partial_warp[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        partial_warp[i_7] = max(partial_warp[i_7], segmented[(((i_7 * 4) + ((rv & 1) * 2)) + (rv >> 1))]);
      }
      partial_warp[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial_warp[i_7]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        ((float*)reduction_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_8 * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 2)) + (((int)threadIdx.x) >> 6))] = partial_warp[i_8];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      updated[i_9] = -CUDART_INF_F;
      for (int w = 0; w < 2; ++w) {
        updated[i_9] = max(updated[i_9], ((float*)reduction_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_9 * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 2)) + w)]);
      }
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      updated[i_10] = max(maximum[i_10], updated[i_10]);
      float condval_1;
      if ((updated[i_10] == -CUDART_INF_F)) {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_1 = updated[i_10];
      }
      correction[i_10] = exp2f((maximum[i_10] - condval_1));
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_11 * 2));
        float condval_2;
        if ((updated[(i_11 & 1)] == -CUDART_INF_F)) {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_2 = updated[(i_11 & 1)];
        }
        float2 v__1 = make_float2(condval_2, condval_2);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_11 * 2)) = __1;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      *(float2*)(segmented + (i_12 * 2)) = *(float2*)(scores + (((i_12 & 1) * 4) + ((i_12 >> 1) * 2)));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      partial_warp[i_13] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        partial_warp[i_13] = (partial_warp[i_13] + segmented[(((i_13 * 4) + ((rv_1 & 1) * 2)) + (rv_1 >> 1))]);
      }
      partial_warp[i_13] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial_warp[i_13]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        ((float*)reduction_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_14 * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 2)) + (((int)threadIdx.x) >> 6)) + 64)] = partial_warp[i_14];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      rowsum[i_15] = 0x0p+0f/*0.000000e+00*/;
      for (int w_1 = 0; w_1 < 2; ++w_1) {
        rowsum[i_15] = (rowsum[i_15] + ((float*)reduction_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 32) + (i_15 * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 2)) + w_1) + 64)]);
      }
    }
    float2 __3;
      float2 v__2 = *(float2*)(denom + 0);
      float2 v__3 = *(float2*)(correction + 0);
      float2 v__4 = *(float2*)(rowsum + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
    *(float2*)(denom + 0) = __3;
    *(float2*)(maximum + 0) = *(float2*)(updated + 0);
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_16 * 2));
        float2 v__6 = make_float2(correction[(i_16 & 1)], correction[(i_16 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_16 * 2)) = __4;
    }
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)scores[0]), ((half_t)scores[1])), __pack_half2(((half_t)scores[2]), ((half_t)scores[3])), __pack_half2(((half_t)scores[4]), ((half_t)scores[5])), __pack_half2(((half_t)scores[6]), ((half_t)scores[7])));
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
        #pragma unroll
        for (int i_17 = 0; i_17 < 2; ++i_17) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((((block & 1) * 2048) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_17) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_17 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 8; ++i_18) {
    *(float2*)(Partial + (((((((((((int)threadIdx.x) & 63) >> 5) * 16384) + ((i_18 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.z) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_18 * 2));
  }
  if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 0) {
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      Maxima[((((((((int)threadIdx.x) & 63) >> 5) * 256) + (i_19 * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = maximum[i_19];
      Sums[((((((((int)threadIdx.x) & 63) >> 5) * 256) + (i_19 * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = denom[i_19];
    }
  }
}

