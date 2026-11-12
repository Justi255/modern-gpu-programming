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
  if (0 < (min(length, ((((int)blockIdx.z) * 26) + 26)) - (((int)blockIdx.z) * 26))) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      if (((((((int)blockIdx.z) * 26) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) < length) && (((i_2 * 4) + (((int)threadIdx.x) >> 5)) < 13)) {
        int physical = Table[((((((int)blockIdx.z) * 13) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) >> 3)];
        *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((int64_t)physical) * (int64_t)16384) + (((((((int64_t)((int)blockIdx.z)) * (int64_t)26) + (((int64_t)i_2) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) & (int64_t)15) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((int64_t)physical) * (int64_t)16384) + (((((((int64_t)((int)blockIdx.z)) * (int64_t)26) + (((int64_t)i_2) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) & (int64_t)15) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
        half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
      }
    }
  }
  for (int block = 0; block < ((max((min(length, ((((int)blockIdx.z) * 26) + 26)) - (((int)blockIdx.z) * 26)), 0) + 31) >> 5); ++block) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((block + 1) < ((max((min(length, ((((int)blockIdx.z) * 26) + 26)) - (((int)blockIdx.z) * 26)), 0) + 31) >> 5)) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 32; ++i_3) {
        ((half_t*)Ks)[((((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_3 >> 3) * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (i_3 & 7))] = half_t(0x0p+0f/*0.000000e+00*/);
        ((half_t*)Vs)[((((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_3 >> 3) * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (i_3 & 7))] = half_t(0x0p+0f/*0.000000e+00*/);
      }
    }
    float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(scores + 0) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
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
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      float condval_1;
      if ((((((((block * 32) + (((int)blockIdx.z) * 26)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) < length) && ((((block * 16) + ((((int)threadIdx.x) >> 5) * 4)) + (((int)threadIdx.x) & 3)) < 13))) {
        condval_1 = (scores[i_4] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
      } else {
        condval_1 = -CUDART_INF_F;
      }
      scores[i_4] = condval_1;
    }
    *(float4*)(segmented + 0) = *(float4*)(scores + 0);
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      float broadcast_var_7 = -CUDART_INF_F;
      partial8_pack[i_5] = make_float2(broadcast_var_7, broadcast_var_7);
      partial8_pack[i_5] = tl::max2(partial8_pack[i_5], *(float2*)(segmented + (i_5 * 2)));
      partial8[i_5] = max((partial8_pack[i_5]).x, (partial8_pack[i_5]).y);
      partial8[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial8[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      partial_warp[i_6] = -CUDART_INF_F;
      partial_warp[i_6] = max(partial_warp[i_6], partial8[i_6]);
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
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_10 * 2));
        float condval_3;
        if ((updated[i_10] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[i_10];
        }
        float2 v__1 = make_float2(condval_3, condval_3);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_10 * 2)) = __1;
    }
    *(float4*)(segmented + 0) = *(float4*)(scores + 0);
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      partial8_pack_1[i_11] = make_float2(broadcast_var_8, broadcast_var_8);
      partial8_pack_1[i_11] = tl::add2(partial8_pack_1[i_11], *(float2*)(segmented + (i_11 * 2)));
      partial8[i_11] = ((partial8_pack_1[i_11]).x + (partial8_pack_1[i_11]).y);
      partial8[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(partial8[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      partial_warp[i_12] = 0x0p+0f/*0.000000e+00*/;
      partial_warp[i_12] = (partial_warp[i_12] + partial8[i_12]);
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
    for (int i_15 = 0; i_15 < 8; ++i_15) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_15 * 2));
        float2 v__6 = make_float2(correction[(i_15 & 1)], correction[(i_15 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_15 * 2)) = __4;
    }
    tl::ptx_stmatrix_m8n8_x2((&(((half_t*)probs)[((((((int)threadIdx.x) & 15) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)scores[0]), ((half_t)scores[1])), __pack_half2(((half_t)scores[2]), ((half_t)scores[3])));
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
        #pragma unroll
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[((((((((block & 1) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_16 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 8; ++i_17) {
    if ((((i_17 & 1) * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
      *(float2*)(Partial + ((((((((i_17 & 1) * 8192) + (((int)blockIdx.y) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_17 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_17 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      if (((i_18 * 2) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        Maxima[((((i_18 * 64) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + ((int)blockIdx.z))] = maximum[i_18];
        Sums[((((i_18 * 64) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + ((int)blockIdx.z))] = denom[i_18];
      }
    }
  }
}

