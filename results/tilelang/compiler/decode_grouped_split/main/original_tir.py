# from tvm.script import tirx as T

@T.prim_func
def kernel(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Lengths_handle: T.handle, Table_handle: T.handle, Out_handle: T.handle, Partial_handle: T.handle, Maxima_handle: T.handle, Sums_handle: T.handle):
    Q = T.match_buffer(Q_handle, (2, 1, 8, 64), "float16", strides=(512, 512, 64, 1))
    K = T.match_buffer(K_handle, (2, 1024, 2, 64), "float16", strides=(131072, 128, 64, 1))
    V = T.match_buffer(V_handle, (2, 1024, 2, 64), "float16", strides=(131072, 128, 64, 1))
    Lengths = T.match_buffer(Lengths_handle, (2,), "int32", strides=(1,))
    Table = T.match_buffer(Table_handle, (1,), "int32", strides=(1,))
    Out = T.match_buffer(Out_handle, (2, 1, 8, 64), "float16", strides=(512, 512, 64, 1))
    Partial = T.match_buffer(Partial_handle, (2, 8, 4, 64), strides=(2048, 256, 64, 1))
    Maxima = T.match_buffer(Maxima_handle, (2, 8, 4), strides=(32, 4, 1))
    Sums = T.match_buffer(Sums_handle, (2, 8, 4), strides=(32, 4, 1))
    # with T.sblock("root"):
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for by in T.thread_binding(4, thread="blockIdx.y"):
            for bz in T.thread_binding(4, thread="blockIdx.z"):
                tx: T.int32 = T.launch_thread_idx(0)
                ty: T.int32 = T.launch_thread_idx(1)
                tz: T.int32 = T.launch_thread_idx(2)
                with T.sblock("tilelang_root"):
                    T.reads()
                    T.writes()
                    T.sblock_attr({"tl.launch_threads": [128, 1, 1]})
                    Qs = T.sblock_alloc_buffer((16, 64), "float16", scope="shared.dyn")
                    Ks = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
                    Vs = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
                    scores = T.sblock_alloc_buffer((16, 64), scope="local.fragment")
                    probs = T.sblock_alloc_buffer((16, 64), "float16", scope="shared.dyn")
                    acc = T.sblock_alloc_buffer((16, 64), scope="local.fragment")
                    maximum = T.sblock_alloc_buffer((16,), scope="local.fragment")
                    updated = T.sblock_alloc_buffer((16,), scope="local.fragment")
                    denom = T.sblock_alloc_buffer((16,), scope="local.fragment")
                    rowsum = T.sblock_alloc_buffer((16,), scope="local.fragment")
                    correction = T.sblock_alloc_buffer((16,), scope="local.fragment")
                    batch: T.int32 = by // 2
                    kvhead: T.int32 = by % 2
                    length: T.int32 = Lengths[batch]
                    lo: T.int32 = bz * 256
                    stop: T.int32 = T.min(length, lo + 256)
                    T.fill(T.region(acc[0, 0], 2, 16, 64), 0)
                    T.fill(T.region(denom[0], 2, 16), 0)
                    T.fill(T.region(maximum[0], 2, 16), T.infinity("float32") * T.float32(-1.0))
                    for i in T.parallel(16):
                        for z in T.parallel(64):
                            Qs[i, z] = T.if_then_else(i < 4, Q[batch, 0, kvhead * 4 + i, z], T.float16(0.0))
                    for block in T.serial((T.max(stop - lo, 0) + 64 - 1) // 64, annotations={"num_stages": 2}):
                        start: T.int32 = lo + block * 64
                        for j in T.parallel(64):
                            for z in T.parallel(64):
                                if start + j < stop:
                                    Ks[j, z] = K[batch, start + j, kvhead, z]
                                    Vs[j, z] = V[batch, start + j, kvhead, z]
                                else:
                                    Ks[j, z] = T.float16(0.0)
                                    Vs[j, z] = T.float16(0.0)
                        T.fill(T.region(scores[0, 0], 2, 16, 64), 0)
                        T.gemm(T.region(Qs[0, 0], 1, 16, 64), T.region(Ks[0, 0], 1, 64, 64), T.region(scores[0, 0], 3, 16, 64), T.bool(False), T.bool(True), 16, 64, 64, 0, T.bool(False), 0, 0, 0)
                        for i in T.parallel(16):
                            for j in T.parallel(64):
                                valid: T.bool = start + j < stop
                                scores[i, j] = T.if_then_else(valid, scores[i, j] * T.float32(0.18033688011112042), T.infinity("float32") * T.float32(-1.0))
                        T.reduce(T.region(scores[0, 0], 1, 16, 64), T.region(updated[0], 2, 16), "max", 1, T.bool(True))
                        for i in T.parallel(16):
                            updated[i] = T.max(maximum[i], updated[i])
                            correction[i] = T.exp2(maximum[i] - T.if_then_else(updated[i] == T.infinity("float32") * T.float32(-1.0), T.float32(0.0), updated[i]))
                        for i in T.parallel(16):
                            for j in T.parallel(64):
                                scores[i, j] = T.exp2(scores[i, j] - T.if_then_else(updated[i] == T.infinity("float32") * T.float32(-1.0), T.float32(0.0), updated[i]))
                        T.reduce(T.region(scores[0, 0], 1, 16, 64), T.region(rowsum[0], 2, 16), "sum", 1, T.bool(True))
                        for i in T.parallel(16):
                            denom[i] = denom[i] * correction[i] + rowsum[i]
                            maximum[i] = updated[i]
                        for i in T.parallel(16):
                            for z in T.parallel(64):
                                acc[i, z] = acc[i, z] * correction[i]
                        T.copy(T.region(scores[0, 0], 1, 16, 64), T.region(probs[0, 0], 2, 16, 64))
                        T.gemm(T.region(probs[0, 0], 1, 16, 64), T.region(Vs[0, 0], 1, 64, 64), T.region(acc[0, 0], 3, 16, 64), T.bool(False), T.bool(False), 16, 64, 64, 0, T.bool(False), 0, 0, 0)
                    for i in T.parallel(16):
                        for z in T.parallel(64):
                            if i < 4:
                                Partial[batch, kvhead * 4 + i, bz, z] = acc[i, z]
                    for i in T.parallel(16):
                        if i < 4:
                            Maxima[batch, kvhead * 4 + i, bz] = maximum[i]
                            Sums[batch, kvhead * 4 + i, bz] = denom[i]