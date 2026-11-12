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
  void* Vs = ((void*)((char*)buf_dyn_shmem + 10240));
  void* probs = ((void*)((char*)buf_dyn_shmem + 18432));
  void* reduction_shared = ((void*)((char*)buf_dyn_shmem + 19456));
  float acc[8];
  float denom[2];
  float maximum[2];
  float scores[4];
  float segmented[4];
  float2 partial_warp_pack[2];
  float partial_warp[2];
  float updated[2];
  float correction[2];
  float2 partial_warp_pack_1[2];
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
    if (((((int)blockIdx.z) * 512) + 32) <= length) {
      for (int pg = 0; pg < 2; ++pg) {
        int physical = Table[((((int)blockIdx.z) * 32) + pg)];
        tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((int64_t)physical) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
        tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((int64_t)physical) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
      }
      tl::cp_async_commit();
    } else {
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
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
  for (int block = 0; block < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 31) >> 5); ++block) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((block + 1) < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 31) >> 5)) {
      if (((((((int)blockIdx.z) * 512) + (block * 32)) + 64) <= length) && (block <= 14)) {
        for (int pg_1 = 0; pg_1 < 2; ++pg_1) {
          int physical_2 = Table[((((((int)blockIdx.z) * 32) + (block * 2)) + pg_1) + 2)];
          tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((((block + 1) & 1) * 2048) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((int64_t)physical_2) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
          tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((((block + 1) & 1) * 2048) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((int64_t)physical_2) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])));
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          if (((((((((int)blockIdx.z) * 512) + (block * 32)) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) + 32) < length) && (block < 15)) {
            int physical_3 = Table[((((((int)blockIdx.z) * 32) + (block * 2)) + i_2) + 2)];
            *(uint4*)(((half_t*)Ks) + ((((((((block + 1) & 1) * 2048) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((int64_t)physical_3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
            *(uint4*)(((half_t*)Vs) + ((((((((block + 1) & 1) * 2048) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((int64_t)physical_3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
          } else {
            half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Ks) + ((((((((block + 1) & 1) * 2048) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
            half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Vs) + ((((((((block + 1) & 1) * 2048) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
          }
        }
      }
    }
    float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(scores + 0) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    {
      half_t A_local[8];
      half_t B_local[4];
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[(((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)Ks)[(((((((block & 1) * 2048) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float condval_1;
      if ((((((((((int)blockIdx.z) * 512) + (block * 32)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_3 & 1)) < length) && (block < 16))) {
        condval_1 = (scores[i_3] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval_1 = -CUDART_INF_F;
      }
      scores[i_3] = condval_1;
    }
    *(float4*)(segmented + 0) = *(float4*)(scores + 0);
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      float broadcast_var_9 = -CUDART_INF_F;
      partial_warp_pack[i_4] = make_float2(broadcast_var_9, broadcast_var_9);
      partial_warp_pack[i_4] = tl::max2(partial_warp_pack[i_4], *(float2*)(segmented + (i_4 * 2)));
      partial_warp[i_4] = max((partial_warp_pack[i_4]).x, (partial_warp_pack[i_4]).y);
      partial_warp[i_4] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial_warp[i_4]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        ((float*)reduction_shared)[(((i_5 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + (((int)threadIdx.x) >> 5))] = partial_warp[i_5];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      updated[i_6] = -CUDART_INF_F;
      for (int w = 0; w < 4; ++w) {
        updated[i_6] = max(updated[i_6], ((float*)reduction_shared)[(((i_6 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w)]);
      }
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      updated[i_7] = max(maximum[i_7], updated[i_7]);
      float condval_2;
      if ((updated[i_7] == -CUDART_INF_F)) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = updated[i_7];
      }
      correction[i_7] = exp2f((maximum[i_7] - condval_2));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_8 * 2));
        float condval_3;
        if ((updated[i_8] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[i_8];
        }
        float2 v__1 = make_float2(condval_3, condval_3);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_8 * 2)) = __1;
    }
    *(float4*)(segmented + 0) = *(float4*)(scores + 0);
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
      partial_warp_pack_1[i_9] = make_float2(broadcast_var_10, broadcast_var_10);
      partial_warp_pack_1[i_9] = tl::add2(partial_warp_pack_1[i_9], *(float2*)(segmented + (i_9 * 2)));
      partial_warp[i_9] = ((partial_warp_pack_1[i_9]).x + (partial_warp_pack_1[i_9]).y);
      partial_warp[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial_warp[i_9]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        ((float*)reduction_shared)[((((i_10 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + (((int)threadIdx.x) >> 5)) + 64)] = partial_warp[i_10];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      rowsum[i_11] = 0x0p+0f/*0.000000e+00*/;
      for (int w_1 = 0; w_1 < 4; ++w_1) {
        rowsum[i_11] = (rowsum[i_11] + ((float*)reduction_shared)[((((i_11 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w_1) + 64)]);
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
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_12 * 2));
        float2 v__6 = make_float2(correction[(i_12 & 1)], correction[(i_12 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_12 * 2)) = __4;
    }
    tl::ptx_stmatrix_m8n8_x2((&(((half_t*)probs)[((((((int)threadIdx.x) & 15) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)scores[0]), ((half_t)scores[1])), __pack_half2(((half_t)scores[2]), ((half_t)scores[3])));
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((((block & 1) * 2048) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 4; ++i_13) {
    if ((((i_13 & 1) * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
      *(float2*)(Partial + ((((((((i_13 & 1) * 8192) + (((int)blockIdx.y) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.z) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_13 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      if (((i_14 * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        Maxima[((((i_14 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = maximum[i_14];
        Sums[((((i_14 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = denom[i_14];
      }
    }
  }
}

