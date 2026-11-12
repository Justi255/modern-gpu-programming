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

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 4096));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 69632));
  void* probs = ((void*)((char*)buf_dyn_shmem + 135168));
  void* reduction_shared = ((void*)((char*)buf_dyn_shmem + 139264));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 139776));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 140288));
  float acc[16];
  float denom[2];
  float maximum[2];
  float scores[16];
  float segmented[16];
  float2 partial8_pack[8];
  float partial8[8];
  float partial_warp[8];
  float updated[2];
  float correction[2];
  float2 partial8_pack_1[8];
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
    if (((((int)blockIdx.z) * 512) + 128) <= length) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int)blockIdx.z) * 524288) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((int)blockIdx.z) * 524288) + (i_3 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
      }
      tl::cp_async_commit();
    } else {
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        if ((((((int)blockIdx.z) * 512) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) < length) {
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int)blockIdx.z) * 524288) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int)blockIdx.z) * 524288) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
        } else {
          half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
          half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
        }
      }
    }
  }
  for (int block = 0; block < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 127) >> 7); ++block) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((block + 1) < ((max((min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512)), 0) + 127) >> 7)) {
      if (((((((int)blockIdx.z) * 512) + (block * 128)) + 256) <= length) && (block <= 2)) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 16; ++i_5) {
          tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((((((block + 1) & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((((int)blockIdx.z) * 524288) + (block * 131072)) + (i_5 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 16; ++i_6) {
          tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((((((block + 1) & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((((int)blockIdx.z) * 524288) + (block * 131072)) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int i_7 = 0; i_7 < 16; ++i_7) {
          if (((((((((int)blockIdx.z) * 512) + (block * 128)) + (i_7 * 8)) + (((int)threadIdx.x) >> 4)) + 128) < length) && (block < 3)) {
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((((int)blockIdx.z) * 524288) + (block * 131072)) + (i_7 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072));
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((((int)blockIdx.z) * 524288) + (block * 131072)) + (i_7 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072));
          } else {
            half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
            half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
          }
        }
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_8 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    }
    {
      half_t A_local[8];
      half_t B_local[16];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[((((((ki >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((((block & 1) * 16384) + ((ki >> 2) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_9 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_9 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      float condval_1;
      if (((((((((((int)blockIdx.z) * 512) + (block * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_10 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_10 & 1)) < length) && (block < 4))) {
        condval_1 = (scores[i_10] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
      } else {
        condval_1 = -CUDART_INF_F;
      }
      scores[i_10] = condval_1;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      *(float2*)(segmented + (i_11 * 2)) = *(float2*)(scores + (((i_11 & 3) * 4) + ((i_11 >> 2) * 2)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      float broadcast_var_9 = -CUDART_INF_F;
      partial8_pack[i_12] = make_float2(broadcast_var_9, broadcast_var_9);
      partial8_pack[i_12] = tl::max2(partial8_pack[i_12], *(float2*)(segmented + (i_12 * 2)));
      partial8[i_12] = max((partial8_pack[i_12]).x, (partial8_pack[i_12]).y);
      partial8[i_12] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial8[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      partial_warp[i_13] = -CUDART_INF_F;
      partial_warp[i_13] = max(partial_warp[i_13], partial8[i_13]);
      partial_warp[i_13] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(partial_warp[i_13], (&(((float*)workspace_1)[0])));
    }
    if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        *(float4*)(((float*)reduction_shared) + ((i_14 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4))) = *(float4*)(partial_warp + (i_14 * 4));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      updated[i_15] = -CUDART_INF_F;
      for (int w = 0; w < 4; ++w) {
        updated[i_15] = max(updated[i_15], ((float*)reduction_shared)[(((i_15 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w)]);
      }
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      updated[i_16] = max(maximum[i_16], updated[i_16]);
      float condval_2;
      if ((updated[i_16] == -CUDART_INF_F)) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = updated[i_16];
      }
      correction[i_16] = exp2f((maximum[i_16] - condval_2));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 8; ++i_17) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_17 * 2));
        float condval_3;
        if ((updated[(i_17 & 1)] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[(i_17 & 1)];
        }
        float2 v__1 = make_float2(condval_3, condval_3);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_17 * 2)) = __1;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      *(float2*)(segmented + (i_18 * 2)) = *(float2*)(scores + (((i_18 & 3) * 4) + ((i_18 >> 2) * 2)));
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 8; ++i_19) {
      float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
      partial8_pack_1[i_19] = make_float2(broadcast_var_10, broadcast_var_10);
      partial8_pack_1[i_19] = tl::add2(partial8_pack_1[i_19], *(float2*)(segmented + (i_19 * 2)));
      partial8[i_19] = ((partial8_pack_1[i_19]).x + (partial8_pack_1[i_19]).y);
      partial8[i_19] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial8[i_19]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      partial_warp[i_20] = 0x0p+0f/*0.000000e+00*/;
      partial_warp[i_20] = (partial_warp[i_20] + partial8[i_20]);
      partial_warp[i_20] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(partial_warp[i_20], (&(((float*)workspace)[0])));
    }
    if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        *(float4*)(((float*)reduction_shared) + (((i_21 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + 64)) = *(float4*)(partial_warp + (i_21 * 4));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      rowsum[i_22] = 0x0p+0f/*0.000000e+00*/;
      for (int w_1 = 0; w_1 < 4; ++w_1) {
        rowsum[i_22] = (rowsum[i_22] + ((float*)reduction_shared)[((((i_22 * 32) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + w_1) + 64)]);
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
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_23 * 2));
        float2 v__6 = make_float2(correction[(i_23 & 1)], correction[(i_23 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_23 * 2)) = __4;
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 2; ++i_24) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[((((((((int)threadIdx.x) >> 6) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_24) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)scores[(i_24 * 8)]), ((half_t)scores[((i_24 * 8) + 1)])), __pack_half2(((half_t)scores[((i_24 * 8) + 2)]), ((half_t)scores[((i_24 * 8) + 3)])), __pack_half2(((half_t)scores[((i_24 * 8) + 4)]), ((half_t)scores[((i_24 * 8) + 5)])), __pack_half2(((half_t)scores[((i_24 * 8) + 6)]), ((half_t)scores[((i_24 * 8) + 7)])));
    }
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((ki_1 >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
        #pragma unroll
        for (int i_25 = 0; i_25 < 2; ++i_25) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[((((((((block & 1) * 16384) + ((((int)threadIdx.x) >> 6) * 8192)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_25) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_25 * 8)])));
        }
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 8; ++i_26) {
    if ((((i_26 & 1) * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
      *(float2*)(Partial + ((((((((i_26 & 1) * 16384) + (((int)blockIdx.y) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_26 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_26 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      if (((i_27 * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        Maxima[((((i_27 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = maximum[i_27];
        Sums[((((i_27 * 128) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = denom[i_27];
      }
    }
  }
}

