# from tvm.script import tirx as T

@T.prim_func
def kernel(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    A = T.match_buffer(A_handle, (512, 1024), "float16", strides=(1024, 1))
    B = T.match_buffer(B_handle, (1024, 512), "float16", strides=(512, 1))
    C = T.match_buffer(C_handle, (512, 512), "float16", strides=(512, 1))
    # with T.sblock("root"):
    for bx in T.thread_binding(128, thread="blockIdx.x"):
        tx: T.int32 = T.launch_thread_idx(0)
        ty: T.int32 = T.launch_thread_idx(1)
        tz: T.int32 = T.launch_thread_idx(2)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            T.sblock_attr({"tl.launch_threads": [128, 1, 1]})
            a_shared = T.sblock_alloc_buffer((32, 32), "float16", scope="shared.dyn")
            b_shared = T.sblock_alloc_buffer((32, 64), "float16", scope="shared.dyn")
            accumulator = T.sblock_alloc_buffer((32, 64), scope="local.fragment")
            by: T.int32 = bx // 8
            bx_1: T.int32 = bx % 8
            T.fill(T.region(accumulator[0, 0], 2, 32, 64), 0)
            for ko in T.serial(32, annotations={"num_stages": 3}):
                T.copy(T.region(A[by * 32, ko * 32], 1, 32, 32), T.region(a_shared[0, 0], 2, 32, 32))
                T.copy(T.region(B[ko * 32, bx_1 * 64], 1, 32, 64), T.region(b_shared[0, 0], 2, 32, 64))
                T.gemm(T.region(a_shared[0, 0], 1, 32, 32), T.region(b_shared[0, 0], 1, 32, 64), T.region(accumulator[0, 0], 3, 32, 64), T.bool(False), T.bool(False), 32, 64, 32, 0, T.bool(False), 0, 0, 0)
            for i in T.parallel(32):
                for j in T.parallel(64):
                    if i < 32 and (j < 64 and (by * 32 + i < 512 and bx_1 * 64 + j < 512)):
                        C[by * 32 + i, bx_1 * 64 + j] = T.Cast("float16", accumulator[i, j])