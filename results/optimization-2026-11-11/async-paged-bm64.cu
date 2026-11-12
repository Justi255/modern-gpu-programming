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

extern "C" __global__ void kernel_kernel(const bfloat16_t* __restrict__ K, const int* __restrict__ Lengths, bfloat16_t* __restrict__ Out, const bfloat16_t* __restrict__ Q, const int* __restrict__ Table, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const bfloat16_t* __restrict__ K, const int* __restrict__ Lengths, bfloat16_t* __restrict__ Out, const bfloat16_t* __restrict__ Q, const int* __restrict__ Table, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 8192));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 24576));
  float acc[32];
  float denom[2];
  float maximum[2];
  float scores[32];
  float updated[2];
  float correction[2];
  float rowsum[2];
  bfloat16_t probs[32];
  int length = Lengths[0];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(denom + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(maximum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(((bfloat16_t*)Qs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.x) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  __syncthreads();
  if ((0 < length) && (1984 < ((((int)blockIdx.x) * 64) + length))) {
    if ((64 <= length) && (32 <= ((length >> 6) + ((int)blockIdx.x)))) {
      for (int pg = 0; pg < 4; ++pg) {
        int physical = Table[pg];
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)Ks)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((int64_t)physical) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])), ((physical < 1024) && (0 <= physical)));
        tl::cp_async_commit();
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)Vs)[(((((pg * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((int64_t)physical) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])), ((physical < 1024) && (0 <= physical)));
        tl::cp_async_commit();
      }
    } else {
      #pragma unroll
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        if ((((i_2 * 16) + (((int)threadIdx.x) >> 3)) < length) && ((((i_2 * 16) + (((int)threadIdx.x) >> 3)) + 1984) < ((((int)blockIdx.x) * 64) + length))) {
          int physical_1 = Table[i_2];
          bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval;
          if (((physical_1 < 1024) && (0 <= physical_1))) {
            condval = *(uint4*)(K + ((((((int64_t)physical_1) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
          } else {
            condval = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
          }
          *(uint4*)(((bfloat16_t*)Ks) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
          bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_1;
          if (((physical_1 < 1024) && (0 <= physical_1))) {
            condval_1 = *(uint4*)(V + ((((((int64_t)physical_1) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
          } else {
            condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
          }
          *(uint4*)(((bfloat16_t*)Vs) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
        } else {
          bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((bfloat16_t*)Ks) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
          bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((bfloat16_t*)Vs) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
        }
      }
    }
  }
  for (int block = 0; block < ((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 1984)), 2048)) + 63) >> 6); ++block) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((block + 1) < ((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 1984)), 2048)) + 63) >> 6)) {
      if (((((block * 64) + 128) <= length) && (((block * 64) + 2112) <= ((((int)blockIdx.x) * 64) + length))) && (block <= 30)) {
        for (int pg_1 = 0; pg_1 < 4; ++pg_1) {
          int physical_2 = Table[(((block * 4) + pg_1) + 4)];
          tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)Ks)[((((((((block + 1) & 1) * 4096) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((int64_t)physical_2) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])), ((physical_2 < 1024) && (0 <= physical_2)));
          tl::cp_async_commit();
          tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)Vs)[((((((((block + 1) & 1) * 4096) + (pg_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((int64_t)physical_2) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8))])), ((physical_2 < 1024) && (0 <= physical_2)));
          tl::cp_async_commit();
        }
      } else {
        #pragma unroll
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          if (((((((block * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 3)) + 64) < length) && (((((block * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 3)) + 2048) < ((((int)blockIdx.x) * 64) + length))) && (block < 31)) {
            int physical_3 = Table[(((block * 4) + i_3) + 4)];
            bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_2;
            if (((physical_3 < 1024) && (0 <= physical_3))) {
              condval_2 = *(uint4*)(K + ((((((int64_t)physical_3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
            } else {
              condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
            }
            *(uint4*)(((bfloat16_t*)Ks) + ((((((((block + 1) & 1) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
            bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_3;
            if (((physical_3 < 1024) && (0 <= physical_3))) {
              condval_3 = *(uint4*)(V + ((((((int64_t)physical_3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
            } else {
              condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
            }
            *(uint4*)(((bfloat16_t*)Vs) + ((((((((block + 1) & 1) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
          } else {
            bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((bfloat16_t*)Ks) + ((((((((block + 1) & 1) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
            bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((bfloat16_t*)Vs) + ((((((((block + 1) & 1) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
          }
        }
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_4 * 4)) = make_float4(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11);
    }
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[32];
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Qs)[((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Ks)[((((((((block & 1) * 4096) + (i_5 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_5 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      float condval_4;
      if ((((((((block * 64) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) < length) && ((((((block * 64) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) + 1984) < ((((int)blockIdx.x) * 64) + length))) & ((((((block * 64) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) + 2048) <= (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + length)))) {
        condval_4 = (scores[i_6] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval_4 = -CUDART_INF_F;
      }
      scores[i_6] = condval_4;
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      updated[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        updated[i_7] = max(updated[i_7], scores[((((rv & 7) * 4) + (i_7 * 2)) + (rv >> 3))]);
      }
      updated[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(updated[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      updated[i_8] = max(maximum[i_8], updated[i_8]);
      float condval_5;
      if ((updated[i_8] == -CUDART_INF_F)) {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_5 = updated[i_8];
      }
      correction[i_8] = exp2f((maximum[i_8] - condval_5));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_9 * 2));
        float condval_6;
        if ((updated[(i_9 & 1)] == -CUDART_INF_F)) {
          condval_6 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_6 = updated[(i_9 & 1)];
        }
        float2 v__1 = make_float2(condval_6, condval_6);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_9 * 2)) = __1;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      rowsum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        rowsum[i_10] = (rowsum[i_10] + scores[((((rv_1 & 7) * 4) + (i_10 * 2)) + (rv_1 >> 3))]);
      }
      rowsum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(rowsum[i_10]);
    }
    float2 __3;
      float2 v__2 = *(float2*)(denom + 0);
      float2 v__3 = *(float2*)(correction + 0);
      float2 v__4 = *(float2*)(rowsum + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
    *(float2*)(denom + 0) = __3;
    *(float2*)(maximum + 0) = *(float2*)(updated + 0);
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_11 * 2));
        float2 v__6 = make_float2(correction[(i_11 & 1)], correction[(i_11 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_11 * 2)) = __4;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      uint2 __5;
      float4 v__7 = *(float4*)(scores + (i_12 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__7))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__7))[1]);
      *(uint2*)(probs + (i_12 * 4)) = __5;
    }
    {
      bfloat16_t B_local_1[32];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)Vs)[(((((((block & 1) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_13 * 8)])));
        }
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 16; ++i_14) {
    bfloat16_t Out_local_cast[2];
    uint1 __6;
    float2 __7;
      float2 v__8 = *(float2*)(acc + (i_14 * 2));
      float2 v__9 = make_float2(max(denom[(i_14 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/), max(denom[(i_14 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/));
      __7.x = (v__8.x/v__9.x);
      __7.y = (v__8.y/v__9.y);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&__7))[0]);
    *(uint1*)(Out_local_cast + 0) = __6;
    *(uint1*)(Out + (((((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((i_14 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((i_14 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(Out_local_cast + 0);
  }
}

