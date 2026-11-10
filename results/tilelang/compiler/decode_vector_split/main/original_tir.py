# from tvm.script import tirx as T

@T.prim_func
def kernel(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Lengths_handle: T.handle, Table_handle: T.handle, Out_handle: T.handle, Partial_handle: T.handle, Maxima_handle: T.handle, Sums_handle: T.handle):
    Q = T.match_buffer(Q_handle, (2, 8, 64), "float16", strides=(512, 64, 1))
    K = T.match_buffer(K_handle, (2, 1024, 2, 64), "float16", strides=(131072, 128, 64, 1))
    V = T.match_buffer(V_handle, (2, 1024, 2, 64), "float16", strides=(131072, 128, 64, 1))
    Lengths = T.match_buffer(Lengths_handle, (2,), "int32", strides=(1,))
    Table = T.match_buffer(Table_handle, (1,), "int32", strides=(1,))
    Out = T.match_buffer(Out_handle, (2, 8, 64), "float16", strides=(512, 64, 1))
    Partial = T.match_buffer(Partial_handle, (2, 8, 4, 64), strides=(2048, 256, 64, 1))
    Maxima = T.match_buffer(Maxima_handle, (2, 8, 4), strides=(32, 4, 1))
    Sums = T.match_buffer(Sums_handle, (2, 8, 4), strides=(32, 4, 1))
    # with T.sblock("root"):
    for bx in T.thread_binding(8, thread="blockIdx.x"):
        for by in T.thread_binding(2, thread="blockIdx.y"):
            for bz in T.thread_binding(4, thread="blockIdx.z"):
                tx: T.int32 = T.launch_thread_idx(0)
                ty: T.int32 = T.launch_thread_idx(1)
                tz: T.int32 = T.launch_thread_idx(2)
                with T.sblock("tilelang_root"):
                    T.reads()
                    T.writes()
                    T.sblock_attr({"tl.launch_threads": [128, 1, 1]})
                    query = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    values = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                    scores = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    p = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    acc = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    weighted = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    maximum = T.sblock_alloc_buffer((1,), scope="local.fragment")
                    updated = T.sblock_alloc_buffer((1,), scope="local.fragment")
                    denom = T.sblock_alloc_buffer((1,), scope="local.fragment")
                    psum = T.sblock_alloc_buffer((1,), scope="local.fragment")
                    alpha = T.sblock_alloc_buffer((1,), scope="local.fragment")
                    lo: T.int32 = bz * 256
                    stop: T.int32 = T.min(Lengths[by], lo + 256)
                    for z in T.parallel(64):
                        query[z] = T.Cast("float32", Q[by, bx, z])
                    T.fill(T.region(acc[0], 2, 64), 0)
                    T.fill(T.region(denom[0], 2, 1), 0)
                    T.fill(T.region(maximum[0], 2, 1), T.infinity("float32") * T.float32(-1.0))
                    for block in range((T.max(stop - lo, 0) + 63) // 64):
                        start: T.int32 = lo + block * 64
                        for j in T.parallel(64):
                            for z in T.parallel(64):
                                if start + j < stop:
                                    values[j, z] = T.Cast("float32", K[by, start + j, bx // 4, z]) * query[z]
                                else:
                                    values[j, z] = T.float32(0.0)
                        T.reduce(T.region(values[0, 0], 1, 64, 64), T.region(scores[0], 2, 64), "sum", 1, T.bool(True))
                        for j in T.parallel(64):
                            scores[j] = T.if_then_else(start + j < stop, scores[j] * T.float32(0.18033688011112042), T.infinity("float32") * T.float32(-1.0))
                        T.reduce(T.region(scores[0], 1, 64), T.region(updated[0], 2, 1), "max", 0, T.bool(True))
                        updated[0] = T.max(maximum[0], updated[0])
                        alpha[0] = T.exp2(maximum[0] - updated[0])
                        maximum[0] = updated[0]
                        for j in T.parallel(64):
                            p[j] = T.exp2(scores[j] - maximum[0])
                        T.reduce(T.region(p[0], 1, 64), T.region(psum[0], 2, 1), "sum", 0, T.bool(True))
                        for j in T.parallel(64):
                            for z in T.parallel(64):
                                if start + j < stop:
                                    values[j, z] = p[j] * T.Cast("float32", V[by, start + j, bx // 4, z])
                                else:
                                    values[j, z] = T.float32(0.0)
                        T.reduce(T.region(values[0, 0], 1, 64, 64), T.region(weighted[0], 2, 64), "sum", 0, T.bool(True))
                        for z in T.parallel(64):
                            acc[z] = acc[z] * alpha[0] + weighted[z]
                        denom[0] = denom[0] * alpha[0] + psum[0]
                    for z in T.parallel(64):
                        Partial[by, bx, bz, z] = acc[z]
                    Maxima[by, bx, bz] = maximum[0]
                    Sums[by, bx, bz] = denom[0]