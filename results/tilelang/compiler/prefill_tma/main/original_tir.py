# from tvm.script import tirx as T

@T.prim_func
def kernel(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Lengths_handle: T.handle, Table_handle: T.handle, Out_handle: T.handle, Partial_handle: T.handle, Maxima_handle: T.handle, Sums_handle: T.handle):
    Q = T.match_buffer(Q_handle, (1, 128, 8, 64), "float16", strides=(65536, 512, 64, 1))
    K = T.match_buffer(K_handle, (1, 256, 2, 64), "float16", strides=(32768, 128, 64, 1))
    V = T.match_buffer(V_handle, (1, 256, 2, 64), "float16", strides=(32768, 128, 64, 1))
    Lengths = T.match_buffer(Lengths_handle, (1,), "int32", strides=(1,))
    Table = T.match_buffer(Table_handle, (1,), "int32", strides=(1,))
    Out = T.match_buffer(Out_handle, (1, 128, 8, 64), "float16", strides=(65536, 512, 64, 1))
    Partial = T.match_buffer(Partial_handle, (1,), strides=(1,))
    Maxima = T.match_buffer(Maxima_handle, (1,), strides=(1,))
    Sums = T.match_buffer(Sums_handle, (1,), strides=(1,))
    # with T.sblock("root"):
    for bx in T.thread_binding(2, thread="blockIdx.x"):
        for by in T.thread_binding(8, thread="blockIdx.y"):
            for bz in T.thread_binding(1, thread="blockIdx.z"):
                tx: T.int32 = T.launch_thread_idx(0)
                ty: T.int32 = T.launch_thread_idx(1)
                tz: T.int32 = T.launch_thread_idx(2)
                with T.sblock("tilelang_root"):
                    T.reads()
                    T.writes()
                    T.sblock_attr({"tl.launch_threads": [128, 1, 1]})
                    Qs = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
                    Ks = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
                    Vs = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
                    scores = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                    probs = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
                    acc = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
                    maximum = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    updated = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    denom = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    rowsum = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    correction = T.sblock_alloc_buffer((64,), scope="local.fragment")
                    batch: T.int32 = by // 8
                    head: T.int32 = by % 8
                    kvhead: T.int32 = head // 4
                    length: T.int32 = Lengths[batch]
                    lo: T.int32 = bz * 256
                    stop: T.int32 = T.min(length, lo + 256)
                    stop_1: T.int32 = T.max(0, T.min(stop, (bx + 1) * 64 + length - 128))
                    T.fill(T.region(acc[0, 0], 2, 64, 64), 0)
                    T.fill(T.region(denom[0], 2, 64), 0)
                    T.fill(T.region(maximum[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                    T.copy(T.region(Q[batch, bx * 64, head, 0], 1, 1, 64, 1, 64), T.region(Qs[0, 0], 2, 64, 64), prefer_instruction="tma")
                    for block in T.serial((T.max(stop_1 - lo, 0) + 64 - 1) // 64, annotations={"num_stages": 2}):
                        start: T.int32 = lo + block * 64
                        T.copy(T.region(K[batch, start, kvhead, 0], 1, 1, 64, 1, 64), T.region(Ks[0, 0], 2, 64, 64), prefer_instruction="tma")
                        for j in T.parallel(64):
                            for z in T.parallel(64):
                                if start + j < stop_1:
                                    Vs[j, z] = V[batch, start + j, kvhead, z]
                                else:
                                    Vs[j, z] = T.float16(0.0)
                        T.fill(T.region(scores[0, 0], 2, 64, 64), 0)
                        T.gemm(T.region(Qs[0, 0], 1, 64, 64), T.region(Ks[0, 0], 1, 64, 64), T.region(scores[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 0, T.bool(False), 0, 0, 0)
                        for i in T.parallel(64):
                            for j in T.parallel(64):
                                valid: T.bool = start + j < stop_1
                                valid_1: T.bool = T.bitwise_and(valid, start + j <= bx * 64 + i + length - 128)
                                scores[i, j] = T.if_then_else(valid_1, scores[i, j] * T.float32(0.18033688011112042), T.infinity("float32") * T.float32(-1.0))
                        T.reduce(T.region(scores[0, 0], 1, 64, 64), T.region(updated[0], 2, 64), "max", 1, T.bool(True))
                        for i in T.parallel(64):
                            updated[i] = T.max(maximum[i], updated[i])
                            correction[i] = T.exp2(maximum[i] - T.if_then_else(updated[i] == T.infinity("float32") * T.float32(-1.0), T.float32(0.0), updated[i]))
                        for i in T.parallel(64):
                            for j in T.parallel(64):
                                scores[i, j] = T.exp2(scores[i, j] - T.if_then_else(updated[i] == T.infinity("float32") * T.float32(-1.0), T.float32(0.0), updated[i]))
                        T.reduce(T.region(scores[0, 0], 1, 64, 64), T.region(rowsum[0], 2, 64), "sum", 1, T.bool(True))
                        for i in T.parallel(64):
                            denom[i] = denom[i] * correction[i] + rowsum[i]
                            maximum[i] = updated[i]
                        for i in T.parallel(64):
                            for z in T.parallel(64):
                                acc[i, z] = acc[i, z] * correction[i]
                        T.copy(T.region(scores[0, 0], 1, 64, 64), T.region(probs[0, 0], 2, 64, 64))
                        T.gemm(T.region(probs[0, 0], 1, 64, 64), T.region(Vs[0, 0], 1, 64, 64), T.region(acc[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 0, T.bool(False), 0, 0, 0)
                    for i in T.parallel(64):
                        for z in T.parallel(64):
                            if bx * 64 + i < 128:
                                Out[batch, bx * 64 + i, head, z] = T.Cast("float16", acc[i, z] / T.max(denom[i], T.float32(1.0000000000000001e-30)))