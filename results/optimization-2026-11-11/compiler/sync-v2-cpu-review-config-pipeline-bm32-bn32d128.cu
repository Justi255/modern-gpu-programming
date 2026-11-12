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
  void* Ks = ((void*)((char*)buf_dyn_shmem + 8192));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 24576));
  void* probs = ((void*)((char*)buf_dyn_shmem + 40960));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 43008));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 44032));
  float acc[32];
  float denom[2];
  float maximum[2];
  float scores[8];
  uint ordered_scores[8];
  uint ordered_max[2];
  float updated[2];
  float correction[2];
  float rowsum[2];
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
    *(uint4*)(((half_t*)Qs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((i_1 * 1024) + (((int)threadIdx.x) * 8)));
  }
  if (0 < (min(length, ((((int)blockIdx.z) * 512) + 512)) - (((int)blockIdx.z) * 512))) {
    if (((((int)blockIdx.z) * 512) + 32) <= length) {
      for (int pg = 0; pg < 2; ++pg) {
        int physical = Table[((((int)blockIdx.z) * 32) + pg)];
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (pg * 1024)) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((int64_t)physical) * (int64_t)2048) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (pg * 1024)) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((int64_t)physical) * (int64_t)2048) + (((int64_t)i_3) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
        }
      }
      tl::cp_async_commit();
    } else {
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        if ((((((int)blockIdx.z) * 512) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) < length) {
          int physical_1 = Table[((((int)blockIdx.z) * 32) + (i_4 >> 1))];
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((int64_t)physical_1) * (int64_t)2048) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((int64_t)physical_1) * (int64_t)2048) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        } else {
          half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
          half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
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
            tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (pg_1 * 1024)) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((int64_t)physical_2) * (int64_t)2048) + (((int64_t)i_5) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
          }
          #pragma unroll
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (pg_1 * 1024)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((int64_t)physical_2) * (int64_t)2048) + (((int64_t)i_6) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))])));
          }
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          if (((((((((int)blockIdx.z) * 512) + (block * 32)) + (i_7 * 8)) + (((int)threadIdx.x) >> 4)) + 32) < length) && (block < 15)) {
            int physical_3 = Table[((((((int)blockIdx.z) * 32) + (block * 2)) + (i_7 >> 1)) + 2)];
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((int64_t)physical_3) * (int64_t)2048) + ((((int64_t)i_7) & (int64_t)1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((int64_t)physical_3) * (int64_t)2048) + ((((int64_t)i_7) & (int64_t)1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
          } else {
            half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
            half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
          }
        }
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_8 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
    }
    {
      half_t A_local[8];
      half_t B_local[8];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[(((((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[(((((((((block & 1) * 4096) + ((ki >> 2) * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      float condval;
      if (((((((((((int)blockIdx.z) * 512) + (block * 32)) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_9 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1)) < length) && (block < 16))) {
        condval = (scores[i_9] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
      } else {
        condval = -CUDART_INF_F;
      }
      scores[i_9] = condval;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      float v_ = scores[i_10];
      uint bits = (*(uint *)(&(v_)));
      uint condval_1;
      if (((bits & (uint)2147483648) != (uint)0)) {
        condval_1 = (bits ^ (uint)4294967295);
      } else {
        condval_1 = (bits ^ (uint)2147483648);
      }
      ordered_scores[i_10] = condval_1;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      ordered_max[i_11] = (uint)0;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        ordered_max[i_11] = max(ordered_max[i_11], ordered_scores[((((rv & 1) * 4) + (i_11 * 2)) + (rv >> 1))]);
      }
    }
    tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<128>, 2, 128>::run_batch((&(ordered_max[0])), (&(((uint*)workspace_1)[0])));
    tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>, 2, 128>::run_batch((&(ordered_max[0])));
    for (int i_s = 0; i_s < 2; ++i_s) {
      uint key = ordered_max[i_s];
      uint condval_2;
      if (((key & (uint)2147483648) != (uint)0)) {
        condval_2 = (key ^ (uint)2147483648);
      } else {
        condval_2 = (key ^ (uint)4294967295);
      }
      uint _reinterpret_tmp = condval_2;
      updated[i_s] = (*(float *)(&(_reinterpret_tmp)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      updated[i_12] = max(maximum[i_12], updated[i_12]);
      float condval_3;
      if ((updated[i_12] == -CUDART_INF_F)) {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_3 = updated[i_12];
      }
      correction[i_12] = exp2f((maximum[i_12] - condval_3));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      float2 __1;
      float2 __2;
        float2 v__1 = *(float2*)(scores + (i_13 * 2));
        float condval_4;
        if ((updated[(i_13 & 1)] == -CUDART_INF_F)) {
          condval_4 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_4 = updated[(i_13 & 1)];
        }
        float2 v__2 = make_float2(condval_4, condval_4);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v__1.x)), *(float2*)(&(v__2.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_13 * 2)) = __1;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      rowsum[i_14] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        rowsum[i_14] = (rowsum[i_14] + scores[((((rv_1 & 1) * 4) + (i_14 * 2)) + (rv_1 >> 1))]);
      }
    }
    tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<128>, 2, 128>::run_batch((&(rowsum[0])), (&(((float*)workspace)[0])));
    tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>, 2, 128>::run_batch((&(rowsum[0])));
    float2 __3;
      float2 v__3 = *(float2*)(denom + 0);
      float2 v__4 = *(float2*)(correction + 0);
      float2 v__5 = *(float2*)(rowsum + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
    *(float2*)(denom + 0) = __3;
    *(float2*)(maximum + 0) = *(float2*)(updated + 0);
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      float2 __4;
        float2 v__6 = *(float2*)(acc + (i_15 * 2));
        float2 v__7 = make_float2(correction[(i_15 & 1)], correction[(i_15 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
      *(float2*)(acc + (i_15 * 2)) = __4;
    }
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((half_t)scores[0]), ((half_t)scores[1])), __pack_half2(((half_t)scores[2]), ((half_t)scores[3])), __pack_half2(((half_t)scores[4]), ((half_t)scores[5])), __pack_half2(((half_t)scores[6]), ((half_t)scores[7])));
    __syncthreads();
    {
      half_t A_local_1[8];
      half_t B_local_1[32];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
        #pragma unroll
        for (int i_16 = 0; i_16 < 4; ++i_16) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[((((((((block & 1) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_16 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 16; ++i_17) {
    *(float2*)(Partial + (((((((((((int)threadIdx.x) & 63) >> 5) * 32768) + ((i_17 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + ((i_17 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_17 * 2));
  }
  if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 6)) == 0) {
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      Maxima[((((((((int)threadIdx.x) & 63) >> 5) * 256) + (i_18 * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = maximum[i_18];
      Sums[((((((((int)threadIdx.x) & 63) >> 5) * 256) + (i_18 * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((int)blockIdx.z))] = denom[i_18];
    }
  }
}

