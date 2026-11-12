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

extern "C" __global__ void kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, half_t* __restrict__ Out, const half_t* __restrict__ Q, const int* __restrict__ Table, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const half_t* __restrict__ K, const int* __restrict__ Lengths, half_t* __restrict__ Out, const half_t* __restrict__ Q, const int* __restrict__ Table, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Ks = ((void*)((char*)buf_dyn_shmem + 0));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 32768));
  float acc[64];
  float denom[2];
  float maximum[2];
  half_t Qs[64];
  float scores[32];
  float updated[2];
  float2 updated_pack[1];
  float correction[2];
  float rowsum[2];
  float2 rowsum_pack[1];
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
  if ((0 < length) && (65 < ((((int)blockIdx.x) * 64) + length))) {
    if ((64 <= length) && (129 <= ((((int)blockIdx.x) * 64) + length))) {
      for (int pg = 0; pg < 4; ++pg) {
        int physical = Table[pg];
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          tl::cp_async_gs<16>((&(((half_t*)Ks)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (pg * 1024)) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int64_t)physical) * (int64_t)16384) + (((int64_t)i_2) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          tl::cp_async_gs<16>((&(((half_t*)Vs)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (pg * 1024)) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((int64_t)physical) * (int64_t)16384) + (((int64_t)i_3) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
        }
      }
      tl::cp_async_commit();
    } else {
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        if ((((i_4 * 8) + (((int)threadIdx.x) >> 4)) < length) && ((((i_4 * 8) + (((int)threadIdx.x) >> 4)) + 65) < ((((int)blockIdx.x) * 64) + length))) {
          int physical_1 = Table[(i_4 >> 1)];
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int64_t)physical_1) * (int64_t)16384) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int64_t)physical_1) * (int64_t)16384) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
          half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
        }
      }
    }
  }
  for (int block = 0; block < ((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 65)), 208)) + 63) >> 6); ++block) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((block + 1) < ((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 65)), 208)) + 63) >> 6)) {
      if (((((block * 64) + 128) <= length) && (((block * 64) + 193) <= ((((int)blockIdx.x) * 64) + length))) && (block <= 1)) {
        for (int pg_1 = 0; pg_1 < 4; ++pg_1) {
          int physical_2 = Table[(((block * 4) + pg_1) + 4)];
          #pragma unroll
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            tl::cp_async_gs<16>((&(((half_t*)Ks)[((((((((((block + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (pg_1 * 1024)) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int64_t)physical_2) * (int64_t)16384) + (((int64_t)i_5) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
          }
          #pragma unroll
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            tl::cp_async_gs<16>((&(((half_t*)Vs)[((((((((((block + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (pg_1 * 1024)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((int64_t)physical_2) * (int64_t)16384) + (((int64_t)i_6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])));
          }
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int i_7 = 0; i_7 < 8; ++i_7) {
          if (((((((block * 64) + (i_7 * 8)) + (((int)threadIdx.x) >> 4)) + 64) < length) && (((((block * 64) + (i_7 * 8)) + (((int)threadIdx.x) >> 4)) + 129) < ((((int)blockIdx.x) * 64) + length))) && (((block * 4) + (i_7 >> 1)) < 9)) {
            int physical_3 = Table[(((block * 4) + (i_7 >> 1)) + 4)];
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int64_t)physical_3) * (int64_t)16384) + ((((int64_t)i_7) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((int64_t)physical_3) * (int64_t)16384) + ((((int64_t)i_7) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Ks) + (((((((((block + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
            half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)Vs) + (((((((((block + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
          }
        }
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_8 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    }
    {
      half_t B_local[32];
      for (int ki = 0; ki < 8; ++ki) {
        #pragma unroll
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[(((((((((block & 1) * 8192) + ((ki >> 2) * 4096)) + (i_9 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_9 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (j * 8)), reinterpret_cast<const unsigned*>(Qs + (ki * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(Qs + (ki * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    if ((((((block * 64) + 64) <= length) && (((block * 64) + 129) <= ((((int)blockIdx.x) * 64) + length))) && (block <= 2)) && (((block * 64) + 192) <= ((((int)blockIdx.x) * 64) + length))) {
      #pragma unroll
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        float broadcast_var_9 = 0x1.0527dbd87e24dp-3f/*1.275174e-01*/;
        float4 __1;
          float4 v_ = *(float4*)(scores + (i_10 * 4));
          float4 v__1 = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
          *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
        *(float4*)(scores + (i_10 * 4)) = __1;
      }
    } else {
      #pragma unroll
      for (int i_11 = 0; i_11 < 32; ++i_11) {
        float condval_1;
        if (((((((((block * 64) + ((i_11 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_11 & 1)) < length) && ((((((block * 64) + ((i_11 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_11 & 1)) + 65) < ((((int)blockIdx.x) * 64) + length))) && (((block * 4) + (i_11 >> 3)) < 13)) & ((((((block * 64) + ((i_11 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_11 & 1)) + 129) <= (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_11 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + length)))) {
          condval_1 = (scores[i_11] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
        } else {
          condval_1 = -CUDART_INF_F;
        }
        scores[i_11] = condval_1;
      }
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      updated[i_12] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        updated[i_12] = max(updated[i_12], scores[((((rv & 7) * 4) + (i_12 * 2)) + (rv >> 3))]);
      }
    }
    updated_pack[0] = float2(updated[0], updated[1]);
    tl::AllReduce<tl::MaxOp_f32x2, 4, 1, 0, tl::NamedBarrier<128>, 1, 128>::run_batch((&(updated_pack[0])));
    updated[0] = (updated_pack[0]).x;
    updated[1] = (updated_pack[0]).y;
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
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      float2 __2;
      float2 __3;
        float2 v__2 = *(float2*)(scores + (i_14 * 2));
        float condval_3;
        if ((updated[(i_14 & 1)] == -CUDART_INF_F)) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = updated[(i_14 & 1)];
        }
        float2 v__3 = make_float2(condval_3, condval_3);
        *(float2*)(&(__3.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      __2.x = exp2f(__3.x);
      __2.y = exp2f(__3.y);
      *(float2*)(scores + (i_14 * 2)) = __2;
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      rowsum[i_15] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        rowsum[i_15] = (rowsum[i_15] + scores[((((rv_1 & 7) * 4) + (i_15 * 2)) + (rv_1 >> 3))]);
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
    for (int i_16 = 0; i_16 < 32; ++i_16) {
      float2 __5;
        float2 v__7 = *(float2*)(acc + (i_16 * 2));
        float2 v__8 = make_float2(correction[(i_16 & 1)], correction[(i_16 & 1)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
      *(float2*)(acc + (i_16 * 2)) = __5;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 8; ++i_17) {
      uint2 __6;
      float4 v__9 = *(float4*)(scores + (i_17 * 4));
      ((half2*)(&__6))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__6))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(probs + (i_17 * 4)) = __6;
    }
    {
      half_t B_local_1[64];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_18 = 0; i_18 < 8; ++i_18) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[((((((((block & 1) * 8192) + ((i_18 >> 2) * 4096)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_18 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_18 * 8)])));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(probs + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 32; ++i_19) {
    half_t Out_local_cast[2];
    if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 129) {
      uint1 __7;
      float2 __8;
        float2 v__10 = *(float2*)(acc + (i_19 * 2));
        float2 v__11 = make_float2(max(denom[(i_19 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/), max(denom[(i_19 & 1)], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/));
        __8.x = (v__10.x/v__11.x);
        __8.y = (v__10.y/v__11.y);
      ((half2*)(&__7))[0] = __float22half2_rn(((float2*)(&__8))[0]);
      *(uint1*)(Out_local_cast + 0) = __7;
      *(uint1*)(Out + (((((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_19 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((i_19 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(Out_local_cast + 0);
    }
  }
}

