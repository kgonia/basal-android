#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define FULL __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define FULL
#endif

// Experiments for gemm_bench (its argument list and layouts). TN = 8 columns per work item throughout.

// A: weights already half (scale folded in), half4 per (k/4, row); 4 rows x 8 columns; half accumulation
FULL kernel void f16w_4x8(global const half4 * w, global const half * unused, __read_only image1d_buffer_t src1,
                          global float * dst, int k, int m, int n, int n_no_padding, ulong offsetd) {
    const int n_4 = n >> 2, gy = get_global_id(0), row0 = get_global_id(1) * 4, bidx = gy * 2;
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0, B;
    global const half4 * wp = w + row0;
    for (int i = 0; i < k; i += 4) {
        const half4 w0 = wp[(i / 4) * m], w1 = wp[(i / 4) * m + 1], w2 = wp[(i / 4) * m + 2], w3 = wp[(i / 4) * m + 3];
#define S(J) B.s0123 = read_imageh(src1, bidx + (i + J) * n_4); B.s4567 = read_imageh(src1, bidx + (i + J) * n_4 + 1); \
        c0 += B * w0.s##J; c1 += B * w1.s##J; c2 += B * w2.s##J; c3 += B * w3.s##J;
        S(0) S(1) S(2) S(3)
#undef S
    }
    dst = (global float *)((global char *)dst + offsetd);
#define ST(c) if (gy * 8 + c < n_no_padding) vstore4((float4)(c0.s##c, c1.s##c, c2.s##c, c3.s##c), 0, dst + (gy * 8 + c) * m + row0);
    ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7)
#undef ST
}

// B: q8_0 like the original, but the 4 rows' bytes for one k are gathered with a byte permute of the 4 uints
// (as_uchar16 of the uint4, then a fixed swizzle) instead of 16 separate extractions
FULL kernel void q8_4x8_swz(global const uint * src0_q, global const half * src0_d, __read_only image1d_buffer_t src1,
                            global float * dst, int k, int m, int n, int n_no_padding, ulong offsetd) {
    const int n_4 = n >> 2, gy = get_global_id(0), row0 = get_global_id(1) * 4, bidx = gy * 2;
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0, B;
    global const uint * wp = src0_q + row0;
    global const half * sp = src0_d + row0;
    for (int i = 0; i < k; i += 4) {
        const char16 p = as_char16(vload4(0, wp + (i / 4) * m));
        const half4 sc = vload4(0, sp + (i / 32) * m);
        // p = r0k0 r0k1 r0k2 r0k3 r1k0 ... ; k-major: (r0k0 r1k0 r2k0 r3k0) (r0k1 ...) ...
        const half16 wq = convert_half16(p.s048C159D26AE37BF) * (half16)(sc, sc, sc, sc);
#define S(J, W) B.s0123 = read_imageh(src1, bidx + (i + J) * n_4); B.s4567 = read_imageh(src1, bidx + (i + J) * n_4 + 1); \
        c0 += B * W.s0; c1 += B * W.s1; c2 += B * W.s2; c3 += B * W.s3;
        S(0, wq.s0123) S(1, wq.s4567) S(2, wq.s89ab) S(3, wq.scdef)
#undef S
    }
    dst = (global float *)((global char *)dst + offsetd);
#define ST(c) if (gy * 8 + c < n_no_padding) vstore4((float4)(c0.s##c, c1.s##c, c2.s##c, c3.s##c), 0, dst + (gy * 8 + c) * m + row0);
    ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7)
#undef ST
}

// C: the original's loop with the weights constant (no weight loads): B reads + FMAs only
FULL kernel void q8_noW(global const uint * src0_q, global const half * src0_d, __read_only image1d_buffer_t src1,
                        global float * dst, int k, int m, int n, int n_no_padding, ulong offsetd) {
    const int n_4 = n >> 2, gy = get_global_id(0), row0 = get_global_id(1) * 4, bidx = gy * 2;
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0, B;
    const half4 w = (half4)(0.01h, 0.02h, 0.03h, 0.04h) * (half)(row0 & 7);
    for (int i = 0; i < k; i += 4) {
#define S(J) B.s0123 = read_imageh(src1, bidx + (i + J) * n_4); B.s4567 = read_imageh(src1, bidx + (i + J) * n_4 + 1); \
        c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;
        S(0) S(1) S(2) S(3)
#undef S
    }
    dst = (global float *)((global char *)dst + offsetd);
#define ST(c) if (gy * 8 + c < n_no_padding) vstore4((float4)(c0.s##c, c1.s##c, c2.s##c, c3.s##c), 0, dst + (gy * 8 + c) * m + row0);
    ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7)
#undef ST
}

// D: the original's loop with B constant (no image reads): weight loads + dequant + FMAs only
FULL kernel void q8_noB(global const uint * src0_q, global const half * src0_d, __read_only image1d_buffer_t src1,
                        global float * dst, int k, int m, int n, int n_no_padding, ulong offsetd) {
    const int gy = get_global_id(0), row0 = get_global_id(1) * 4;
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    const half8 B = (half8)(0.1h) * (half)(gy & 7);
    global const uint * wp = src0_q + row0;
    global const half * sp = src0_d + row0;
    for (int i = 0; i < k; i += 4) {
        const uint4 pack4 = vload4(0, wp + (i / 4) * m);
        const half4 scale = vload4(0, sp + (i / 32) * m);
        const char4 p0 = as_char4(pack4.s0), p1 = as_char4(pack4.s1), p2 = as_char4(pack4.s2), p3 = as_char4(pack4.s3);
#define S(J) { const half4 wj = convert_half4((char4)(p0.s##J, p1.s##J, p2.s##J, p3.s##J)) * scale; \
        c0 += B * wj.s0; c1 += B * wj.s1; c2 += B * wj.s2; c3 += B * wj.s3; }
        S(0) S(1) S(2) S(3)
#undef S
    }
    dst = (global float *)((global char *)dst + offsetd);
#define ST(c) if (gy * 8 + c < n_no_padding) vstore4((float4)(c0.s##c, c1.s##c, c2.s##c, c3.s##c), 0, dst + (gy * 8 + c) * m + row0);
    ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7)
#undef ST
}

// E: the original's FMA pattern with neither weight loads nor image reads (B and w vary per k from registers)
FULL kernel void q8_noWB(global const uint * src0_q, global const half * src0_d, __read_only image1d_buffer_t src1,
                         global float * dst, int k, int m, int n, int n_no_padding, ulong offsetd) {
    const int gy = get_global_id(0), row0 = get_global_id(1) * 4;
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B = (half8)(0.1h) * (half)(gy & 7);
    half4 w = (half4)(0.01h, 0.02h, 0.03h, 0.04h) * (half)(row0 & 7);
    for (int i = 0; i < k; i += 4) {
#define S(J) c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3; B = B * (half)0.999h;
        S(0) S(1) S(2) S(3)
#undef S
        w = w * (half)1.001h;
    }
    dst = (global float *)((global char *)dst + offsetd);
#define ST(c) if (gy * 8 + c < n_no_padding) vstore4((float4)(c0.s##c, c1.s##c, c2.s##c, c3.s##c), 0, dst + (gy * 8 + c) * m + row0);
    ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7)
#undef ST
}

// F: the original with the activations read from the plain buffer: with local size (1, L) every lane of a subgroup
// has the same gy, so the address is uniform across it
FULL kernel void q8_bbuf(global const uint * src0_q, global const half * src0_d, global const half8 * src1,
                         global float * dst, int k, int m, int n, int n_no_padding, ulong offsetd) {
    const int n_8 = n >> 3, gy = get_global_id(0), row0 = get_global_id(1) * 4;
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    global const uint * wp = src0_q + row0;
    global const half * sp = src0_d + row0;
    global const half8 * bp = src1 + gy;
    for (int i = 0; i < k; i += 4) {
        const uint4 pack4 = vload4(0, wp + (i / 4) * m);
        const half4 scale = vload4(0, sp + (i / 32) * m);
        const char4 p0 = as_char4(pack4.s0), p1 = as_char4(pack4.s1), p2 = as_char4(pack4.s2), p3 = as_char4(pack4.s3);
#define S(J) { const half8 B = bp[(i + J) * n_8]; \
        const half4 wj = convert_half4((char4)(p0.s##J, p1.s##J, p2.s##J, p3.s##J)) * scale; \
        c0 += B * wj.s0; c1 += B * wj.s1; c2 += B * wj.s2; c3 += B * wj.s3; }
        S(0) S(1) S(2) S(3)
#undef S
    }
    dst = (global float *)((global char *)dst + offsetd);
#define ST(c) if (gy * 8 + c < n_no_padding) vstore4((float4)(c0.s##c, c1.s##c, c2.s##c, c3.s##c), 0, dst + (gy * 8 + c) * m + row0);
    ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7)
#undef ST
}
