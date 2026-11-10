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

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, half_t* __restrict__ Out, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, half_t* __restrict__ Out, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 4096));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 12288));
  void* probs = ((void*)((char*)buf_dyn_shmem + 20480));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 25088));
  float acc[16];
  float denom[2];
  float maximum[2];
  float scores[16];
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
    *(uint4*)(((half_t*)Qs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.x) * 16384) + (i_1 * 8192)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  for (int block = 0; block < ((max(0, min(min(length, (((((int)blockIdx.x) * 32) + length) - 96)), 256)) + 63) >> 6); ++block) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      if (((((block * 64) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < length) && (((((block * 64) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) + 96) < ((((int)blockIdx.x) * 32) + length))) {
        *(uint4*)(((half_t*)Ks) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((block * 8192) + (i_2 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) >> 2) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
        *(uint4*)(((half_t*)Vs) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((block * 8192) + (i_2 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) >> 2) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      } else {
        half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Ks) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
        half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Vs) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
    }
    {
      half_t A_local[8];
      half_t B_local[16];
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((((int)threadIdx.x) >> 6) * 2048) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      float condval;
      if (((((((((block * 64) + ((((int)threadIdx.x) >> 6) * 32)) + ((i_5 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) < length) && (((((((block * 64) + ((((int)threadIdx.x) >> 6) * 32)) + ((i_5 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) + 96) < ((((int)blockIdx.x) * 32) + length))) & (((((((block * 64) + ((((int)threadIdx.x) >> 6) * 32)) + ((i_5 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) + 128) <= (((((((int)blockIdx.x) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + (((i_5 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + length)))) {
        condval = (scores[i_5] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval = -CUDART_INF_F;
      }
      scores[i_5] = condval;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      updated[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        updated[i_6] = max(updated[i_6], scores[((((rv & 3) * 4) + (i_6 * 2)) + (rv >> 2))]);
      }
      __syncthreads();
      updated[i_6] = tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<128>>::run(updated[i_6], (&(((float*)workspace)[0])));
      updated[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(updated[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      updated[i_7] = max(maximum[i_7], updated[i_7]);
      float condval_1;
      if ((updated[i_7] == -CUDART_INF_F)) {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_1 = updated[i_7];
      }
      correction[i_7] = exp2f((maximum[i_7] - condval_1));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_8 * 2));
        float condval_2;
        if ((updated[(i_8 & 1)] == -CUDART_INF_F)) {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_2 = updated[(i_8 & 1)];
        }
        float2 v__1 = make_float2(condval_2, condval_2);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_8 * 2)) = __1;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      rowsum[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        rowsum[i_9] = (rowsum[i_9] + scores[((((rv_1 & 3) * 4) + (i_9 * 2)) + (rv_1 >> 2))]);
      }
      __syncthreads();
      rowsum[i_9] = tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<128>>::run(rowsum[i_9], (&(((float*)workspace_1)[0])));
      rowsum[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(rowsum[i_9]);
    }
    float2 __3;
      float2 v__2 = *(float2*)(denom + 0);
      float2 v__3 = *(float2*)(correction + 0);
      float2 v__4 = *(float2*)(rowsum + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
    *(float2*)(denom + 0) = __3;
    *(float2*)(maximum + 0) = *(float2*)(updated + 0);
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_10 * 2));
        float2 v__6 = make_float2(correction[(i_10 & 1)], correction[(i_10 & 1)]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_10 * 2)) = __4;
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_11) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)scores[(i_11 * 8)]), ((half_t)scores[((i_11 * 8) + 1)])), __pack_half2(((half_t)scores[((i_11 * 8) + 2)]), ((half_t)scores[((i_11 * 8) + 3)])), __pack_half2(((half_t)scores[((i_11 * 8) + 4)]), ((half_t)scores[((i_11 * 8) + 5)])), __pack_half2(((half_t)scores[((i_11 * 8) + 6)]), ((half_t)scores[((i_11 * 8) + 7)])));
    }
    {
      half_t A_local_1[8];
      half_t B_local_1[16];
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)probs)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
        #pragma unroll
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((ki_1 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_12) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_12 * 8)])));
        }
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 8; ++i_13) {
    half_t Out_local_cast[2];
    uint1 __5;
    float2 __6;
      float2 v__7 = *(float2*)(acc + (i_13 * 2));
      float2 v__8 = make_float2(max(denom[(i_13 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/), max(denom[(i_13 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/));
      __6.x = (v__7.x/v__8.x);
      __6.y = (v__7.y/v__8.y);
    ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
    *(uint1*)(Out_local_cast + 0) = __5;
    *(uint1*)(Out + ((((((((((int)blockIdx.x) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + ((i_13 & 1) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(Out_local_cast + 0);
  }
}

