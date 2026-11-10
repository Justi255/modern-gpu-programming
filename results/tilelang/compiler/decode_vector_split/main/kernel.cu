#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
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
  float query[1];
  float acc[1];
  float denom[1];
  float maximum[1];
  float values[64];
  float scores[64];
  extern __shared__ __align__(1024) float workspace[];
  float2 updated_pack[1];
  float updated[1];
  float alpha[1];
  float p[64];
  float2 psum_pack[1];
  float psum[1];
  float2 weighted_pack[1];
  float weighted[1];
  int stop = min(Lengths[((int)blockIdx.y)], ((((int)blockIdx.z) * 256) + 256));
  query[0] = ((float)Q[(((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63))]);
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  denom[0] = 0x0p+0f/*0.000000e+00*/;
  maximum[0] = -CUDART_INF_F;
  for (int block = 0; block < ((max((stop - (((int)blockIdx.z) * 256)), 0) + 63) >> 6); ++block) {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      if ((((((int)blockIdx.z) * 256) + (block * 64)) + i) < stop) {
        half_t condval;
        if ((((block >> 2) + ((int)blockIdx.z)) < 4)) {
          condval = K[((((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + (((int64_t)block) * (int64_t)8192)) + (((int64_t)i) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) & (int64_t)63))];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        values[i] = (((float)condval) * query[0]);
      } else {
        values[i] = 0x0p+0f/*0.000000e+00*/;
      }
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      scores[i_1] = 0x0p+0f/*0.000000e+00*/;
      scores[i_1] = (scores[i_1] + values[i_1]);
      __syncthreads();
      scores[i_1] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<128>>::run(scores[i_1], (&(workspace[0])));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      float condval_1;
      if (((((((int)blockIdx.z) * 256) + (block * 64)) + i_2) < stop)) {
        condval_1 = (scores[i_2] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval_1 = -CUDART_INF_F;
      }
      scores[i_2] = condval_1;
    }
    float broadcast_var = -CUDART_INF_F;
    updated_pack[0] = make_float2(broadcast_var, broadcast_var);
    #pragma unroll
    for (int rv = 0; rv < 32; ++rv) {
      updated_pack[0] = tl::max2(updated_pack[0], *(float2*)(scores + (rv * 2)));
    }
    updated[0] = max((updated_pack[0]).x, (updated_pack[0]).y);
    updated[0] = max(maximum[0], updated[0]);
    alpha[0] = exp2f((maximum[0] - updated[0]));
    maximum[0] = updated[0];
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      float4 __1;
      float4 __2;
        float4 v_ = *(float4*)(scores + (i_3 * 4));
        float4 v__1 = make_float4(maximum[0], maximum[0], maximum[0], maximum[0]);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
        *(float2*)(&(__2.z)) = tl::sub2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      __1.z = exp2f(__2.z);
      __1.w = exp2f(__2.w);
      *(float4*)(p + (i_3 * 4)) = __1;
    }
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    psum_pack[0] = make_float2(broadcast_var_1, broadcast_var_1);
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
      psum_pack[0] = tl::add2(psum_pack[0], *(float2*)(p + (rv_1 * 2)));
    }
    psum[0] = ((psum_pack[0]).x + (psum_pack[0]).y);
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      if ((((((int)blockIdx.z) * 256) + (block * 64)) + i_4) < stop) {
        half_t condval_2;
        if ((((block >> 2) + ((int)blockIdx.z)) < 4)) {
          condval_2 = V[((((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + (((int64_t)block) * (int64_t)8192)) + (((int64_t)i_4) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) & (int64_t)63))];
        } else {
          condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        values[i_4] = (p[i_4] * ((float)condval_2));
      } else {
        values[i_4] = 0x0p+0f/*0.000000e+00*/;
      }
    }
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    weighted_pack[0] = make_float2(broadcast_var_2, broadcast_var_2);
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
      weighted_pack[0] = tl::add2(weighted_pack[0], *(float2*)(values + (rv_2 * 2)));
    }
    weighted[0] = ((weighted_pack[0]).x + (weighted_pack[0]).y);
    acc[0] = ((acc[0] * alpha[0]) + weighted[0]);
    denom[0] = ((denom[0] * alpha[0]) + psum[0]);
  }
  if ((((int)threadIdx.x) >> 6) == 0) {
    Partial[((((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 256)) + (((int)blockIdx.z) * 64)) + (((int)threadIdx.x) & 63))] = acc[0];
  }
  Maxima[(((((int)blockIdx.y) * 32) + (((int)blockIdx.x) * 4)) + ((int)blockIdx.z))] = maximum[0];
  Sums[(((((int)blockIdx.y) * 32) + (((int)blockIdx.x) * 4)) + ((int)blockIdx.z))] = denom[0];
}

