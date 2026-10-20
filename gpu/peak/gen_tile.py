"""Write OpenCL q8_0 x half prefill GEMM kernels with a TM x TN register tile per work item, for gemm_bench.

    python gen_tile.py <out dir>     (one q8_t<TM>x<TN>[f].cl per kernel: the Adreno compiler rejects a whole program
                                      when one kernel in it is too large)

Kernel q8_t<TM>x<TN>[f]: TM rows (4 or 8, from one or two uint4 of the noshuffle layout) x TN columns (8 or 16,
two or four half4 texels of the transposed activations per k). Suffix f: half sums per 32-wide q8_0 block, float
sums across blocks. Arguments and layouts as kernel_gemm_noshuffle_q8_0_f32.
"""
import sys

HEAD = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define FULL __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define FULL
#endif
"""


def kernel(tm, tn, blockf):
    nb = tn // 8  # half8 column groups
    rows, groups = range(tm), range(nb)
    name = f"q8_t{tm}x{tn}{'f' if blockf else ''}"
    o = [f"FULL kernel void {name}(global const uint * src0_q, global const half * src0_d,",
         "        __read_only image1d_buffer_t src1, global float * dst, int k, int m, int n, int n_no_padding, ulong offsetd) {",
         f"    const int n_4 = n >> 2, gy = get_global_id(0), row0 = get_global_id(1) * {tm}, bidx = gy * {tn // 4};",
         "    global const uint * wp = src0_q + row0;",
         "    global const half * sp = src0_d + row0;"]
    o.append("    half8 " + ", ".join(f"c{r}_{g} = 0" for r in rows for g in groups) + ";")
    if blockf:
        o.append("    float8 " + ", ".join(f"f{r}_{g} = 0" for r in rows for g in groups) + ";")
    o.append("    for (int ib = 0; ib < k; ib += 32) {")
    for h in range(tm // 4):
        o.append(f"        const half4 s{h} = vload4({h}, sp + (ib / 32) * m);")
    o.append("        for (int i = ib; i < ib + 32; i += 4) {")
    for h in range(tm // 4):
        o.append(f"            const uint4 p{h} = vload4({h}, wp + (i / 4) * m);")
        for j in range(4):
            o.append(f"            const char4 a{h}{j} = as_char4(p{h}.s{j});")
    for j in range(4):
        for g in groups:
            o.append(f"            half8 B{g}_{j}; B{g}_{j}.s0123 = read_imageh(src1, bidx + {2 * g} + (i + {j}) * n_4);"
                     f" B{g}_{j}.s4567 = read_imageh(src1, bidx + {2 * g + 1} + (i + {j}) * n_4);")
        for h in range(tm // 4):
            o.append(f"            const half4 w{h}_{j} = convert_half4((char4)(a{h}0.s{j}, a{h}1.s{j}, a{h}2.s{j}, a{h}3.s{j})) * s{h};")
        for r in rows:
            for g in groups:
                o.append(f"            c{r}_{g} += B{g}_{j} * w{r // 4}_{j}.s{r % 4};")
    o.append("        }")
    if blockf:
        for r in rows:
            for g in groups:
                o.append(f"        f{r}_{g} += convert_float8(c{r}_{g}); c{r}_{g} = 0;")
    o.append("    }")
    o.append("    dst = (global float *)((global char *)dst + offsetd);")
    acc = (lambda r, g: f"f{r}_{g}") if blockf else (lambda r, g: f"convert_float8(c{r}_{g})")
    for r in rows:
        for g in groups:
            o.append(f"    const float8 r{r}_{g} = {acc(r, g)};")
    for c in range(tn):
        g, l = divmod(c, 8)
        vals = ", ".join(f"r{r}_{g}.s{l}" for r in rows)
        o.append(f"    if (gy * {tn} + {c} < n_no_padding) vstore{tm}((float{tm})({vals}), 0, dst + (gy * {tn} + {c}) * m + row0);")
    o.append("}")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    import os
    for tm in (4, 8):
        for tn in (8, 16):
            for blockf in (False, True):
                name = f"q8_t{tm}x{tn}{'f' if blockf else ''}"
                with open(os.path.join(sys.argv[1], name + ".cl"), "w", newline="\n") as fh:
                    fh.write(HEAD + "\n" + kernel(tm, tn, blockf))
