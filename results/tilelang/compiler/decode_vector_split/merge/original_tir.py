# from tvm.script import tirx as T

@T.prim_func
def kernel(Partial_handle: T.handle, Maxima_handle: T.handle, Sums_handle: T.handle, Out_handle: T.handle):
    Partial = T.match_buffer(Partial_handle, (2, 8, 4, 64), strides=(2048, 256, 64, 1))
    Maxima = T.match_buffer(Maxima_handle, (2, 8, 4), strides=(32, 4, 1))
    Sums = T.match_buffer(Sums_handle, (2, 8, 4), strides=(32, 4, 1))
    Out = T.match_buffer(Out_handle, (2, 8, 64), "float16", strides=(512, 64, 1))
    # with T.sblock("root"):
    for bx in T.thread_binding(8, thread="blockIdx.x"):
        for by in T.thread_binding(2, thread="blockIdx.y"):
            tx: T.int32 = T.launch_thread_idx(0)
            ty: T.int32 = T.launch_thread_idx(1)
            tz: T.int32 = T.launch_thread_idx(2)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                T.sblock_attr({"tl.launch_threads": [128, 1, 1]})
                maximum = T.sblock_alloc_buffer((4,), scope="local.fragment")
                global_max = T.sblock_alloc_buffer((1,), scope="local.fragment")
                weights = T.sblock_alloc_buffer((4,), scope="local.fragment")
                normalizer = T.sblock_alloc_buffer((1,), scope="local.fragment")
                values = T.sblock_alloc_buffer((4, 64), scope="local.fragment")
                result = T.sblock_alloc_buffer((64,), scope="local.fragment")
                for s in T.parallel(4):
                    maximum[s] = Maxima[by, bx, s]
                T.reduce(T.region(maximum[0], 1, 4), T.region(global_max[0], 2, 1), "max", 0, T.bool(True))
                for s in T.parallel(4):
                    maximum[s] = T.exp2(maximum[s] - T.if_then_else(global_max[0] == T.infinity("float32") * T.float32(-1.0), T.float32(0.0), global_max[0]))
                    weights[s] = maximum[s] * Sums[by, bx, s]
                T.reduce(T.region(weights[0], 1, 4), T.region(normalizer[0], 2, 1), "sum", 0, T.bool(True))
                for s in T.parallel(4):
                    for z in T.parallel(64):
                        values[s, z] = maximum[s] * Partial[by, bx, s, z]
                T.reduce(T.region(values[0, 0], 1, 4, 64), T.region(result[0], 2, 64), "sum", 0, T.bool(True))
                for z in T.parallel(64):
                    Out[by, bx, z] = T.Cast("float16", result[z] / T.max(normalizer[0], T.float32(1.0000000000000001e-30)))