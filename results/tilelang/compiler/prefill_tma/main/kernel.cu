#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(__grid_constant__ const CUtensorMap K_desc, const int* __restrict__ Lengths, half_t* __restrict__ Out, __grid_constant__ const CUtensorMap Q_desc, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(__grid_constant__ const CUtensorMap K_desc, const int* __restrict__ Lengths, half_t* __restrict__ Out, __grid_constant__ const CUtensorMap Q_desc, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 8192));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 24576));
  void* probs = ((void*)((char*)buf_dyn_shmem + 32768));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 40960));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 40960));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 41472));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 41472));
  __shared__ __align__(16) uint64_t Q_to_Qs_mbarrier_mem[4];
  auto Q_to_Qs_mbarrier = reinterpret_cast<Barrier*>(Q_to_Qs_mbarrier_mem);
  float acc[32];
  float denom[4];
  float maximum[4];
  float scores[32];
  float updated[4];
  float correction[4];
  float rowsum[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    Q_to_Qs_mbarrier[0].init(1);
    Q_to_Qs_mbarrier[1].init(1);
    Q_to_Qs_mbarrier[2].init(1);
    Q_to_Qs_mbarrier[3].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  int length = Lengths[0];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(denom + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float4*)(maximum + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  if (tl::tl_shuffle_elect<128>()) {
    Q_to_Qs_mbarrier[0].arrive_and_expect_tx(8192);
    tl::tma_load(Q_desc, Q_to_Qs_mbarrier[0], (&(((half_t*)Qs)[0])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y), 0);
  }
  Q_to_Qs_mbarrier[0].wait(0);
  if ((0 < length) && (64 < ((((int)blockIdx.x) * 64) + length))) {
    if (tl::tl_shuffle_elect<128>()) {
      Q_to_Qs_mbarrier[1].arrive_and_expect_tx(8192);
      tl::tma_load(K_desc, Q_to_Qs_mbarrier[1], (&(((half_t*)Ks)[0])), 0, 0, (((int)blockIdx.y) >> 2), 0);
    }
    Q_to_Qs_mbarrier[1].wait(0);
  }
  if ((64 < length) && (128 < ((((int)blockIdx.x) * 64) + length))) {
    if (tl::tl_shuffle_elect<128>()) {
      Q_to_Qs_mbarrier[2].arrive_and_expect_tx(8192);
      tl::tma_load(K_desc, Q_to_Qs_mbarrier[2], (&(((half_t*)Ks)[4096])), 0, 64, (((int)blockIdx.y) >> 2), 0);
    }
    Q_to_Qs_mbarrier[2].wait(0);
  }
  for (int block = 0; block < (((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) - 2); ++block) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      *(uint4*)(((half_t*)Vs) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((block * 8192) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) >> 2) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_2 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    {
      half_t A_local[16];
      half_t B_local[16];
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((block * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
        }
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    if (tl::tl_shuffle_elect<128>()) {
      Q_to_Qs_mbarrier[3].arrive_and_expect_tx(8192);
      tl::fence_proxy_async();
      tl::tma_load(K_desc, Q_to_Qs_mbarrier[3], (&(((half_t*)Ks)[(block * 4096)])), 0, ((block * 64) + 128), (((int)blockIdx.y) >> 2), 0);
    }
    Q_to_Qs_mbarrier[3].wait(block);
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      float condval;
      if (((((((((block * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_6 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) < length) && (((((((block * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_6 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) + 64) < ((((int)blockIdx.x) * 64) + length))) & (((((((block * 64) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_6 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) + 128) <= ((((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_6 >> 4) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + length)))) {
        condval = (scores[i_6] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
      } else {
        condval = -CUDART_INF_F;
      }
      scores[i_6] = condval;
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      updated[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        updated[i_7] = max(updated[i_7], scores[(((((i_7 >> 1) * 16) + ((rv & 3) * 4)) + ((i_7 & 1) * 2)) + (rv >> 2))]);
      }
      __syncthreads();
      updated[i_7] = tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<128>>::run(updated[i_7], (&(((float*)workspace_2)[0])));
      updated[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(updated[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      updated[i_8] = max(maximum[i_8], updated[i_8]);
      float condval_1;
      if ((updated[i_8] == -CUDART_INF_F)) {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_1 = updated[i_8];
      }
      correction[i_8] = exp2f((maximum[i_8] - condval_1));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      float2 __1;
      float2 __2;
        float2 v_ = *(float2*)(scores + (i_9 * 2));
        float condval_2;
        if ((updated[(((i_9 >> 3) * 2) + (i_9 & 1))] == -CUDART_INF_F)) {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_2 = updated[(((i_9 >> 3) * 2) + (i_9 & 1))];
        }
        float2 v__1 = make_float2(condval_2, condval_2);
        *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(scores + (i_9 * 2)) = __1;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      rowsum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        rowsum[i_10] = (rowsum[i_10] + scores[(((((i_10 >> 1) * 16) + ((rv_1 & 3) * 4)) + ((i_10 & 1) * 2)) + (rv_1 >> 2))]);
      }
      __syncthreads();
      rowsum[i_10] = tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<128>>::run(rowsum[i_10], (&(((float*)workspace_1)[0])));
      rowsum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(rowsum[i_10]);
    }
    float4 __3;
      float4 v__2 = *(float4*)(denom + 0);
      float4 v__3 = *(float4*)(correction + 0);
      float4 v__4 = *(float4*)(rowsum + 0);
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)), *(float2*)(&(v__4.x)));
      *(float2*)(&(__3.z)) = tl::fma2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)), *(float2*)(&(v__4.z)));
    *(float4*)(denom + 0) = __3;
    *(float4*)(maximum + 0) = *(float4*)(updated + 0);
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      float2 __4;
        float2 v__5 = *(float2*)(acc + (i_11 * 2));
        float2 v__6 = make_float2(correction[(((i_11 >> 3) * 2) + (i_11 & 1))], correction[(((i_11 >> 3) * 2) + (i_11 & 1))]);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(acc + (i_11 * 2)) = __4;
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_12 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)scores[(i_12 * 8)]), ((half_t)scores[((i_12 * 8) + 1)])), __pack_half2(((half_t)scores[((i_12 * 8) + 2)]), ((half_t)scores[((i_12 * 8) + 3)])), __pack_half2(((half_t)scores[((i_12 * 8) + 4)]), ((half_t)scores[((i_12 * 8) + 5)])), __pack_half2(((half_t)scores[((i_12 * 8) + 6)]), ((half_t)scores[((i_12 * 8) + 7)])));
    }
    {
      half_t A_local_1[16];
      half_t B_local_1[16];
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[(i_13 * 8)])));
        }
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((ki_1 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_14 * 8)])));
        }
        for (int i_15 = 0; i_15 < 2; ++i_15) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((i_15 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (((i_15 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int block_1 = 0; block_1 < 2; ++block_1) {
    __syncthreads();
    if (2 <= (((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) + block_1)) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        if ((((((((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) * 64) + (block_1 * 64)) + (i_16 * 16)) + (((int)threadIdx.x) >> 3)) < (length + 128)) && (((((((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) * 64) + (block_1 * 64)) + (i_16 * 16)) + (((int)threadIdx.x) >> 3)) < (((((int)blockIdx.x) * 64) + length) + 64))) {
          *(uint4*)(((half_t*)Vs) + (((((i_16 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((((((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) * 8192) + (block_1 * 8192)) + (i_16 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)blockIdx.y) >> 2) * 64)) + ((((int)threadIdx.x) & 7) * 8)) - 16384));
        } else {
          half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
          *(uint4*)(((half_t*)Vs) + (((((i_16 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
        }
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(scores + (i_17 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
      }
    }
    __syncthreads();
    if (2 <= (((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) + block_1)) {
      {
        half_t A_local_2[16];
        half_t B_local_2[16];
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          #pragma unroll
          for (int i_18 = 0; i_18 < 2; ++i_18) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Qs)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_18 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_2[(i_18 * 8)])));
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 2; ++i_19) {
            tl::ptx_ldmatrix_x4((&(((half_t*)Ks)[((((((((((((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) + block_1) & 1) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_19 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_19 * 8)])));
          }
          for (int i_20 = 0; i_20 < 2; ++i_20) {
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + ((i_20 * 16) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + (((i_20 * 16) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 32; ++i_21) {
        float condval_3;
        if ((((((((((((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) * 64) + (block_1 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_21 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_21 & 1)) < (length + 128)) && (((((((((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) * 64) + (block_1 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_21 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_21 & 1)) < (((((int)blockIdx.x) * 64) + length) + 64))) & (((((((((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) * 64) + (block_1 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_21 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_21 & 1)) <= ((((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + ((i_21 >> 4) * 16)) + (((i_21 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + length)))) {
          condval_3 = (scores[i_21] * 0x1.71547652b82fep-3f/*1.803369e-01*/);
        } else {
          condval_3 = -CUDART_INF_F;
        }
        scores[i_21] = condval_3;
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 4; ++i_22) {
        updated[i_22] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
          updated[i_22] = max(updated[i_22], scores[(((((i_22 >> 1) * 16) + ((rv_2 & 3) * 4)) + ((i_22 & 1) * 2)) + (rv_2 >> 2))]);
        }
        __syncthreads();
        updated[i_22] = tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<128>>::run(updated[i_22], (&(((float*)workspace_3)[0])));
        updated[i_22] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(updated[i_22]);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 4; ++i_23) {
        updated[i_23] = max(maximum[i_23], updated[i_23]);
        float condval_4;
        if ((updated[i_23] == -CUDART_INF_F)) {
          condval_4 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_4 = updated[i_23];
        }
        correction[i_23] = exp2f((maximum[i_23] - condval_4));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 16; ++i_24) {
        float2 __5;
        float2 __6;
          float2 v__7 = *(float2*)(scores + (i_24 * 2));
          float condval_5;
          if ((updated[(((i_24 >> 3) * 2) + (i_24 & 1))] == -CUDART_INF_F)) {
            condval_5 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_5 = updated[(((i_24 >> 3) * 2) + (i_24 & 1))];
          }
          float2 v__8 = make_float2(condval_5, condval_5);
          *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
        __5.x = exp2f(__6.x);
        __5.y = exp2f(__6.y);
        *(float2*)(scores + (i_24 * 2)) = __5;
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 4; ++i_25) {
        rowsum[i_25] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
          rowsum[i_25] = (rowsum[i_25] + scores[(((((i_25 >> 1) * 16) + ((rv_3 & 3) * 4)) + ((i_25 & 1) * 2)) + (rv_3 >> 2))]);
        }
        __syncthreads();
        rowsum[i_25] = tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<128>>::run(rowsum[i_25], (&(((float*)workspace)[0])));
        rowsum[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(rowsum[i_25]);
      }
      float4 __7;
        float4 v__9 = *(float4*)(denom + 0);
        float4 v__10 = *(float4*)(correction + 0);
        float4 v__11 = *(float4*)(rowsum + 0);
        *(float2*)(&(__7.x)) = tl::fma2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
        *(float2*)(&(__7.z)) = tl::fma2(*(float2*)(&(v__9.z)), *(float2*)(&(v__10.z)), *(float2*)(&(v__11.z)));
      *(float4*)(denom + 0) = __7;
      *(float4*)(maximum + 0) = *(float4*)(updated + 0);
      #pragma unroll
      for (int i_26 = 0; i_26 < 16; ++i_26) {
        float2 __8;
          float2 v__12 = *(float2*)(acc + (i_26 * 2));
          float2 v__13 = make_float2(correction[(((i_26 >> 3) * 2) + (i_26 & 1))], correction[(((i_26 >> 3) * 2) + (i_26 & 1))]);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__12.x)), *(float2*)(&(v__13.x)));
        *(float2*)(acc + (i_26 * 2)) = __8;
      }
      __syncthreads();
      #pragma unroll
      for (int i_27 = 0; i_27 < 4; ++i_27) {
        tl::ptx_stmatrix_m8n8_x4((&(((half_t*)probs)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_27 >> 1) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((half_t)scores[(i_27 * 8)]), ((half_t)scores[((i_27 * 8) + 1)])), __pack_half2(((half_t)scores[((i_27 * 8) + 2)]), ((half_t)scores[((i_27 * 8) + 3)])), __pack_half2(((half_t)scores[((i_27 * 8) + 4)]), ((half_t)scores[((i_27 * 8) + 5)])), __pack_half2(((half_t)scores[((i_27 * 8) + 6)]), ((half_t)scores[((i_27 * 8) + 7)])));
      }
    }
    __syncthreads();
    if (2 <= (((max(0, min(min(length, (((((int)blockIdx.x) * 64) + length) - 64)), 256)) + 63) >> 6) + block_1)) {
      {
        half_t A_local_3[16];
        half_t B_local_3[16];
        for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
          #pragma unroll
          for (int i_28 = 0; i_28 < 2; ++i_28) {
            tl::ptx_ldmatrix_x4((&(((half_t*)probs)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_28 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_3[(i_28 * 8)])));
          }
          #pragma unroll
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            tl::ptx_ldmatrix_x4_trans((&(((half_t*)Vs)[(((((ki_3 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_29) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_29 * 8)])));
          }
          for (int i_30 = 0; i_30 < 2; ++i_30) {
            for (int j_3 = 0; j_3 < 2; ++j_3) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((i_30 * 16) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_30 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (((i_30 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_30 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 16; ++i_31) {
    half_t Out_local_cast[2];
    uint1 __9;
    float2 __10;
      float2 v__14 = *(float2*)(acc + (i_31 * 2));
      float2 v__15 = make_float2(max(denom[(((i_31 >> 3) * 2) + (i_31 & 1))], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/), max(denom[(((i_31 >> 3) * 2) + (i_31 & 1))], 0x1.4484bfeebc2ap-100f/*1.000000e-30*/));
      __10.x = (v__14.x/v__15.x);
      __10.y = (v__14.y/v__15.y);
    ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&__10))[0]);
    *(uint1*)(Out_local_cast + 0) = __9;
    *(uint1*)(Out + (((((((((((int)blockIdx.x) * 32768) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + ((i_31 >> 3) * 8192)) + ((i_31 & 1) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_31 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(Out_local_cast + 0);
  }
}

