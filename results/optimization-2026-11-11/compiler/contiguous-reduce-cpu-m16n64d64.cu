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
  void* Ks = ((void*)((char*)buf_dyn_shmem + 2048));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 18432));
  void* probs = ((void*)((char*)buf_dyn_shmem + 34816));
  void* reduction_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  float acc[8];
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
  if (0 < (min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512))) {
    if (((((int)blockIdx.z) * 512) + 64) <= length) {
      for (int pg = 0; pg < 4; ++pg) {
        int physical = Table[((((int)blockIdx.z) * 32) + pg)];
        tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((int64_t)physical) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
        tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((int64_t)physical) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
      }
      tl::cp_async_commit();
    } else {
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        if ((((((int)blockIdx.z) * 512) + (i_1 * 16)) + (((int)threadIdx.x) >> 3)) < length) {
          int physical_1 = Table[((((int)blockIdx.z) * 32) + i_1)];
          *(uint4*)(((half_t*)Ks) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((int64_t)physical_1) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
          *(uint4*)(((half_t*)Vs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((int64_t)physical_1) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
        } else {
          half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Ks) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
          half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Vs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
        }
      }
    }
  }
  for (int block = 0; block < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 63) >> 6); ++block) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((block + 1) < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 63) >> 6)) {
      if (((((((int)blockIdx.z) * 512) + (block * 64)) + 128) <= length) && (block <= 6)) {
        for (int pg_1 = 0; pg_1 < 4; ++pg_1) {
          int physical_2 = Table[((((((int)blockIdx.z) * 32) + (block * 4)) + pg_1) + 4)];
          tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((((block + 1) & 1) * 4096) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((int64_t)physical_2) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
          tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((((block + 1) & 1) * 4096) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((int64_t)physical_2) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          if (((((((((int)blockIdx.z) * 512) + (block * 64)) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) + 64) < length) && (block < 7)) {
            int physical_3 = Table[((((((int)blockIdx.z) * 32) + (block * 4)) + i_2) + 4)];
            *(uint4*)(((half_t*)Ks) + ((((((((block + 1) & 1) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((int64_t)physical_3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
            *(uint4*)(((half_t*)Vs) + ((((((((block + 1) & 1) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((int64_t)physical_3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
          } else {
            half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Ks) + ((((((((block + 1) & 1) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
            half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Vs) + ((((((((block + 1) & 1) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
          }
        }
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    }
    {
      half_t A_local[8];
      half_t B_local[8];
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[(((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[(((((((block & 1) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      float condval_1;
      if (((((((((((int)blockIdx.z) * 512) + (block * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) < length) && (block < 8))) {
        condval_1 = (scores[i_4] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval_1 = -CUDART_INF_F;
      }
      scores[i_4] = condval_1;
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(float2*)(segmented + (i_5 * 2)) = *(float2*)(scores + (((i_5 & 1) * 4) + ((i_5 >> 1) * 2)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      partial_warp[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        partial_warp[i_6] = max(partial_warp[i_6], segmented[(((i_6 * 4) + ((rv & 1) * 2)) + (rv >> 1))]);
      }
      partial_warp[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial_warp[i_6]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        ((float*)reduction_shared)[(((i_7 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + (((int)threadIdx.x) >> 5))] = partial_warp[i_7];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      updated[i_8] = -CUDART_INF_F;
      for (int w = 0; w < 4; ++w) {
        updated[i_8] = max(updated[i_8], ((float*)reduction_shared)[(((i_8 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w)]);
      }
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      updated[i_9] = max(maximum[i_9], updated[i_9]);
      float condval_2;
      if ((updated[i_9] == -CUDART_INF_F)) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = updated[i_9];
      }
      correction[i_9] = exp2f((maximum[i_9] - condval_2));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_10 * 2));
        float condval_3;
        if ((updated[(i_10 & 1)] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[(i_10 & 1)];
        }
        float2 v__1 = make_float2(condval_3, condval_3);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_10 * 2)) = __1;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      *(float2*)(segmented + (i_11 * 2)) = *(float2*)(scores + (((i_11 & 1) * 4) + ((i_11 >> 1) * 2)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      partial_warp[i_12] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        partial_warp[i_12] = (partial_warp[i_12] + segmented[(((i_12 * 4) + ((rv_1 & 1) * 2)) + (rv_1 >> 1))]);
      }
      partial_warp[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial_warp[i_12]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        ((float*)reduction_shared)[((((i_13 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + (((int)threadIdx.x) >> 5)) + 64)] = partial_warp[i_13];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      rowsum[i_14] = 0x0p+0f/*0.000000e+00*/;
      for (int w_1 = 0; w_1 < 4; ++w_1) {
        rowsum[i_14] = (rowsum[i_14] + ((float*)reduction_shared)[((((i_14 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w_1) + 64)]);
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
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_15 * 2));
        float2 v__6 = make_float2(correction[(i_15 & 1)], correction[(i_15 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_15 * 2)) = __4;
    }
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[(((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)scores[0]), ((half_t)scores[1])), __pack_half2(((half_t)scores[2]), ((half_t)scores[3])), __pack_half2(((half_t)scores[4]), ((half_t)scores[5])), __pack_half2(((half_t)scores[6]), ((half_t)scores[7])));
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[(((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((((block & 1) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 4; ++i_16) {
    if ((((i_16 & 1) * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
      *(float2*)(Partial + ((((((((i_16 & 1) * 8192) + (((int)blockIdx.y) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.z) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_16 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_16 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      if (((i_17 * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        Maxima[((((i_17 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = maximum[i_17];
        Sums[((((i_17 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = denom[i_17];
      }
    }
  }
}

