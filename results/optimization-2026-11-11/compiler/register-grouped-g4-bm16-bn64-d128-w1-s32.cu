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

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const int* __restrict__ Table, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(32, 1) kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, float* __restrict__ Maxima, float* __restrict__ Partial, const half_t* __restrict__ Q, float* __restrict__ Sums, const int* __restrict__ Table, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 4096));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 20480));
  float acc[64];
  float denom[2];
  float maximum[2];
  float scores[32];
  float updated[2];
  float correction[2];
  float rowsum[2];
  half_t probs[32];
  int length = Lengths[0];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(denom + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(maximum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((i_1 < 2)) {
      condval = *(uint4*)(Q + (((((int)blockIdx.y) * 512) + (i_1 * 256)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
    }
    *(uint4*)(((half_t*)Qs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_1 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  for (int block = 0; block < ((max((min(length, ((((int)blockIdx.z) * 256) + 256)) - (((int)blockIdx.z) * 256)), 0) + 63) >> 6); ++block) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; ++i_2) {
      if ((((((((int)blockIdx.z) * 256) + (block * 64)) + (i_2 * 2)) + (((int)threadIdx.x) >> 4)) < length) && (block < 4)) {
        int physical = Table[(((((int)blockIdx.z) * 16) + (block * 4)) + (i_2 >> 3))];
        half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_1;
        if (((physical < 512) && (0 <= physical))) {
          condval_1 = *(uint4*)(K + (((((((int64_t)physical) * (int64_t)16384) + ((((int64_t)i_2) & (int64_t)7) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_1 = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
        }
        *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
        half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_2;
        if (((physical < 512) && (0 <= physical))) {
          condval_2 = *(uint4*)(V + (((((((int64_t)physical) * (int64_t)16384) + ((((int64_t)i_2) & (int64_t)7) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval_2 = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
        }
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
      } else {
        half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
        half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    }
    {
      half_t A_local[8];
      half_t B_local[32];
      __syncthreads();
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[((((((ki >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_4 = 0; i_4 < 4; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((ki >> 2) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      float condval_3;
      if ((((((((((int)blockIdx.z) * 256) + (block * 64)) + ((i_5 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) < length) && (block < 4))) {
        condval_3 = (scores[i_5] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
      } else {
        condval_3 = -CUDART_INF_F;
      }
      scores[i_5] = condval_3;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      updated[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        updated[i_6] = max(updated[i_6], scores[((((rv & 7) * 4) + (i_6 * 2)) + (rv >> 3))]);
      }
      updated[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(updated[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      updated[i_7] = max(maximum[i_7], updated[i_7]);
      float condval_4;
      if ((updated[i_7] == -CUDART_INF_F)) {
        condval_4 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_4 = updated[i_7];
      }
      correction[i_7] = exp2f((maximum[i_7] - condval_4));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_8 * 2));
        float condval_5;
        if ((updated[(i_8 & 1)] == -CUDART_INF_F)) {
          condval_5 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_5 = updated[(i_8 & 1)];
        }
        float2 v__1 = make_float2(condval_5, condval_5);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_8 * 2)) = __1;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      rowsum[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        rowsum[i_9] = (rowsum[i_9] + scores[((((rv_1 & 7) * 4) + (i_9 * 2)) + (rv_1 >> 3))]);
      }
      rowsum[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<32>>::run(rowsum[i_9]);
    }
    float2 __3;
      float2 v__2 = *(float2*)(denom + 0);
      float2 v__3 = *(float2*)(correction + 0);
      float2 v__4 = *(float2*)(rowsum + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
    *(float2*)(denom + 0) = __3;
    *(float2*)(maximum + 0) = *(float2*)(updated + 0);
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_10 * 2));
        float2 v__6 = make_float2(correction[(i_10 & 1)], correction[(i_10 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_10 * 2)) = __4;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      uint2 __5;
      float4 v__7 = *(float4*)(scores + (i_11 * 4));
      ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&v__7))[0]);
      ((half2*)(&__5))[1] = __float22half2_rn(((float2*)(&v__7))[1]);
      *(uint2*)(probs + (i_11 * 4)) = __5;
    }
    {
      half_t B_local_1[64];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 8; ++i_12) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((((i_12 >> 2) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_12 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_12 * 8)])));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 32; ++i_13) {
    if ((((i_13 & 1) * 2) + (((int)threadIdx.x) >> 4)) < 1) {
      *(float2*)(Partial + (((((((i_13 & 1) * 32768) + (((int)blockIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 4096)) + (((int)blockIdx.z) * 128)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_13 * 2));
    }
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      if (((i_14 * 2) + (((int)threadIdx.x) >> 4)) < 1) {
        Maxima[((((i_14 * 256) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 2) * 32)) + ((int)blockIdx.z))] = maximum[i_14];
        Sums[((((i_14 * 256) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 2) * 32)) + ((int)blockIdx.z))] = denom[i_14];
      }
    }
  }
}

