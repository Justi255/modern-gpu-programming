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

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, half_t* __restrict__ Out, const half_t* __restrict__ Q, const int* __restrict__ Table, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, half_t* __restrict__ Out, const half_t* __restrict__ Q, const int* __restrict__ Table, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Ks = ((void*)((char*)buf_dyn_shmem + 0));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 32768));
  float acc[64];
  float denom[2];
  float maximum[2];
  half_t Qs[64];
  float scores[64];
  float updated[2];
  float2 updated_pack[1];
  float correction[2];
  float rowsum[2];
  float2 rowsum_pack[1];
  half_t probs[64];
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
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
    uint1 condval;
    if ((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 129)) {
      condval = *(uint1*)(Q + (((((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_1 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((i_1 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    } else {
      condval = make_uint1(__pack_half2(broadcast_var_3, broadcast_var_3));
    }
    *(uint1*)(Qs + (i_1 * 2)) = condval;
  }
  for (int block = 0; block < ((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 65)), 208)) + 127) >> 7); ++block) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      if ((((((block * 128) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) < length) && (((((block * 128) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + 65) < ((((int)blockIdx.x) * 64) + length))) && (((block * 8) + (i_2 >> 1)) < 13)) {
        int physical = Table[((block * 8) + (i_2 >> 1))];
        *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int64_t)physical) * (int64_t)16384) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int64_t)physical) * (int64_t)16384) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
        half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    {
      half_t B_local[64];
      __syncthreads();
      for (int ki = 0; ki < 8; ++ki) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 8; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((ki >> 2) * 8192) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
        }
        for (int j = 0; j < 8; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (j * 8)), reinterpret_cast<const unsigned*>(Qs + (ki * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(Qs + (ki * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    if ((((((block * 128) + 128) <= length) && (((block * 128) + 193) <= ((((int)blockIdx.x) * 64) + length))) && (block == 0)) && (((block * 128) + 256) <= ((((int)blockIdx.x) * 64) + length))) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        float broadcast_var_7 = 0x1.0527dbd87e24dp-3f/*1.275174e-01*/;
        float4 __1;
          float4 v_ = *(float4*)(scores + (i_5 * 4));
          float4 v__1 = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
          *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
        *(float4*)(scores + (i_5 * 4)) = __1;
      }
    } else {
      #pragma unroll
      for (int i_6 = 0; i_6 < 64; ++i_6) {
        float condval_1;
        if (((((((((block * 128) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) < length) && ((((((block * 128) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) + 65) < ((((int)blockIdx.x) * 64) + length))) && (((block * 8) + (i_6 >> 3)) < 13)) & ((((((block * 128) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) + 129) <= (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + length)))) {
          condval_1 = (scores[i_6] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
        } else {
          condval_1 = -CUDART_INF_F;
        }
        scores[i_6] = condval_1;
      }
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      updated[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 32; ++rv) {
        updated[i_7] = max(updated[i_7], scores[((((rv & 15) * 4) + (i_7 * 2)) + (rv >> 4))]);
      }
    }
    updated_pack[0] = float2(updated[0], updated[1]);
    tl::AllReduce<tl::MaxOp_f32x2, 4, 1, 0, tl::NamedBarrier<128>, 1, 128>::run_batch((&(updated_pack[0])));
    updated[0] = (updated_pack[0]).x;
    updated[1] = (updated_pack[0]).y;
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      updated[i_8] = max(maximum[i_8], updated[i_8]);
      float condval_2;
      if ((updated[i_8] == -CUDART_INF_F)) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = updated[i_8];
      }
      correction[i_8] = exp2f((maximum[i_8] - condval_2));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      float2 __2;
      float2 __3;
        float2 v__2 = *(float2*)(scores + (i_9 * 2));
        float condval_3;
        if ((updated[(i_9 & 1)] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[(i_9 & 1)];
        }
        float2 v__3 = make_float2(condval_3, condval_3);
        *(float2*)(&(__3.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      __2.x = exp2f(__3.x);
      __2.y = exp2f(__3.y);
      *(float2*)(scores + (i_9 * 2)) = __2;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      rowsum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        rowsum[i_10] = (rowsum[i_10] + scores[((((rv_1 & 15) * 4) + (i_10 * 2)) + (rv_1 >> 4))]);
      }
    }
    rowsum_pack[0] = float2(rowsum[0], rowsum[1]);
    tl::AllReduce<tl::SumOp_f32x2, 4, 1, 0, tl::NamedBarrier<128>, 1, 128>::run_batch((&(rowsum_pack[0])));
    rowsum[0] = (rowsum_pack[0]).x;
    rowsum[1] = (rowsum_pack[0]).y;
    float2 __4;
      float2 v__4 = *(float2*)(denom + 0);
      float2 v__5 = *(float2*)(correction + 0);
      float2 v__6 = *(float2*)(rowsum + 0);
      *(float2*)(&(__4.x)) = tl::fma2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
    *(float2*)(denom + 0) = __4;
    *(float2*)(maximum + 0) = *(float2*)(updated + 0);
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      float2 __5;
        float2 v__7 = *(float2*)(acc + (i_11 * 2));
        float2 v__8 = make_float2(correction[(i_11 & 1)], correction[(i_11 & 1)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
      *(float2*)(acc + (i_11 * 2)) = __5;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      uint2 __6;
      float4 v__9 = *(float4*)(scores + (i_12 * 4));
      ((half2*)(&__6))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__6))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(probs + (i_12 * 4)) = __6;
    }
    {
      half_t B_local_1[64];
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        #pragma unroll
        for (int i_13 = 0; i_13 < 8; ++i_13) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((((i_13 >> 2) * 8192) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_13 * 8)])));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 32; ++i_14) {
    half_t Out_local_cast[2];
    if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 129) {
      uint1 __7;
      float2 __8;
        float2 v__10 = *(float2*)(acc + (i_14 * 2));
        float2 v__11 = make_float2(max(denom[(i_14 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/), max(denom[(i_14 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/));
        __8.x = (v__10.x/v__11.x);
        __8.y = (v__10.y/v__11.y);
      ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&__8))[0]);
      *(uint1*)(Out_local_cast + 0) = __7;
      *(uint1*)(Out + (((((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_14 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((i_14 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(Out_local_cast + 0);
    }
  }
}

