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
  void* Vs = ((void*)((char*)buf_dyn_shmem + 20480));
  void* probs = ((void*)((char*)buf_dyn_shmem + 36864));
  void* reduction_shared = ((void*)((char*)buf_dyn_shmem + 37888));
  float acc[16];
  float denom[2];
  float maximum[2];
  float scores[4];
  float segmented[4];
  float2 partial8_pack[2];
  float partial8[2];
  float partial_warp[2];
  float updated[2];
  float correction[2];
  float2 partial8_pack_1[2];
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
    half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((i_1 * 2) + (((int)threadIdx.x) >> 6)) < 1)) {
      condval = *(uint4*)(Q + (((i_1 * 1024) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
    }
    *(uint4*)(((half_t*)Qs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  if (0 < (min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512))) {
    if (((((int)blockIdx.z) * 512) + 32) <= length) {
      for (int pg = 0; pg < 2; ++pg) {
        int physical = Table[((((int)blockIdx.z) * 32) + pg)];
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (pg * 1024)) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int64_t)physical) * (int64_t)16384) + (((int64_t)i_2) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (pg * 1024)) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((int64_t)physical) * (int64_t)16384) + (((int64_t)i_3) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
        }
      }
      tl::cp_async_commit();
    } else {
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        if ((((((int)blockIdx.z) * 512) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) < length) {
          int physical_1 = Table[((((int)blockIdx.z) * 32) + (i_4 >> 1))];
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int64_t)physical_1) * (int64_t)16384) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int64_t)physical_1) * (int64_t)16384) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
          half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
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
          #pragma unroll
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (pg_1 * 1024)) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int64_t)physical_2) * (int64_t)16384) + (((int64_t)i_5) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
          }
          #pragma unroll
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (pg_1 * 1024)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((int64_t)physical_2) * (int64_t)16384) + (((int64_t)i_6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
          }
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          if (((((((((int)blockIdx.z) * 512) + (block * 32)) + (i_7 * 8)) + (((int)threadIdx.x) >> 4)) + 32) < length) && (block < 15)) {
            int physical_3 = Table[((((((int)blockIdx.z) * 32) + (block * 2)) + (i_7 >> 1)) + 2)];
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int64_t)physical_3) * (int64_t)16384) + ((((int64_t)i_7) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int64_t)physical_3) * (int64_t)16384) + ((((int64_t)i_7) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
            half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
          }
        }
      }
    }
    float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(scores + 0) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    {
      half_t A_local[8];
      half_t B_local[4];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[((((((ki >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((half_t*)Ks)[((((((((block & 1) * 4096) + ((ki >> 2) * 2048)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      float condval_1;
      if ((((((((((int)blockIdx.z) * 512) + (block * 32)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_8 & 1)) < length) && (block < 16))) {
        condval_1 = (scores[i_8] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
      } else {
        condval_1 = -CUDART_INF_F;
      }
      scores[i_8] = condval_1;
    }
    *(float4*)(segmented + 0) = *(float4*)(scores + 0);
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      float broadcast_var_9 = -CUDART_INF_F;
      partial8_pack[i_9] = make_float2(broadcast_var_9, broadcast_var_9);
      partial8_pack[i_9] = tl::max2(partial8_pack[i_9], *(float2*)(segmented + (i_9 * 2)));
      partial8[i_9] = max((partial8_pack[i_9]).x, (partial8_pack[i_9]).y);
      partial8[i_9] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial8[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      partial_warp[i_10] = -CUDART_INF_F;
      partial_warp[i_10] = max(partial_warp[i_10], partial8[i_10]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        ((float*)reduction_shared)[(((i_11 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + (((int)threadIdx.x) >> 5))] = partial_warp[i_11];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      updated[i_12] = -CUDART_INF_F;
      for (int w = 0; w < 4; ++w) {
        updated[i_12] = max(updated[i_12], ((float*)reduction_shared)[(((i_12 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w)]);
      }
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      updated[i_13] = max(maximum[i_13], updated[i_13]);
      float condval_2;
      if ((updated[i_13] == -CUDART_INF_F)) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = updated[i_13];
      }
      correction[i_13] = exp2f((maximum[i_13] - condval_2));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_14 * 2));
        float condval_3;
        if ((updated[i_14] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[i_14];
        }
        float2 v__1 = make_float2(condval_3, condval_3);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_14 * 2)) = __1;
    }
    *(float4*)(segmented + 0) = *(float4*)(scores + 0);
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
      partial8_pack_1[i_15] = make_float2(broadcast_var_10, broadcast_var_10);
      partial8_pack_1[i_15] = tl::add2(partial8_pack_1[i_15], *(float2*)(segmented + (i_15 * 2)));
      partial8[i_15] = ((partial8_pack_1[i_15]).x + (partial8_pack_1[i_15]).y);
      partial8[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial8[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      partial_warp[i_16] = 0x0p+0f/*0.000000e+00*/;
      partial_warp[i_16] = (partial_warp[i_16] + partial8[i_16]);
    }
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        ((float*)reduction_shared)[((((i_17 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + (((int)threadIdx.x) >> 5)) + 64)] = partial_warp[i_17];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      rowsum[i_18] = 0x0p+0f/*0.000000e+00*/;
      for (int w_1 = 0; w_1 < 4; ++w_1) {
        rowsum[i_18] = (rowsum[i_18] + ((float*)reduction_shared)[((((i_18 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w_1) + 64)]);
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
    for (int i_19 = 0; i_19 < 8; ++i_19) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_19 * 2));
        float2 v__6 = make_float2(correction[(i_19 & 1)], correction[(i_19 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_19 * 2)) = __4;
    }
    tl::ptx_stmatrix_m8n8_x2((&(((half_t*)probs)[((((((int)threadIdx.x) & 15) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)scores[0]), ((half_t)scores[1])), __pack_half2(((half_t)scores[2]), ((half_t)scores[3])));
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
        #pragma unroll
        for (int i_20 = 0; i_20 < 2; ++i_20) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[((((((((block & 1) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_20) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_20 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 8; ++i_21) {
    if ((((i_21 & 1) * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
      *(float2*)(Partial + ((((((((i_21 & 1) * 16384) + (((int)blockIdx.y) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_21 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_21 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      if (((i_22 * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        Maxima[((((i_22 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = maximum[i_22];
        Sums[((((i_22 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = denom[i_22];
      }
    }
  }
}

