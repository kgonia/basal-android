// gemm_bench: one prefill GEMM of llama.cpp's Adreno q8_0 layouts in isolation, for trying kernels quickly: it builds
// <kernel file> with <options>, runs <kernel name> on random weights and activations and prints its TFLOPS and its
// largest error against a CPU reference.
//
//   gemm_bench <file.cl> <kernel> <M> <K> <N> <rows>[x<columns>] per work item <local0> <local1> [options...]
//
// Layouts (those of kernel_gemm_noshuffle_q8_0_f32): weights q8_0 "noshuffle", q as one uint per (k/4, row) holding
// 4 consecutive k of that row, d as one half per (k/32, row); or, with -DF16W, w as one half4 per (k/4, row) with the
// scale folded in. Activations: half, transposed to [k][n] with n padded, read as an RGBA half image1d_buffer
// (or, with -DBBUF, as the buffer itself). Output: float, dst[col * M + row]. Global size (N / columns, M / rows) per work item; columns
// default to 8, and N is padded to a multiple of them.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s: %d\n", #x, e_); exit(1); } } while (0)

static uint16_t to_half(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7c00);
    return (uint16_t) (sign | (e << 10) | ((m + 0x1000) >> 13));
}
static float from_half(uint16_t h) {
    const uint32_t sign = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x = e == 0 ? sign : sign | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

int main(int argc, char ** argv) {
    if (argc < 9) { fprintf(stderr, "usage: gemm_bench <file.cl> <kernel> <M> <K> <N> <tm> <local0> <local1> [opts]\n"); return 2; }
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf(); const std::string src = ss.str();
    const char * kname = argv[2];
    const int M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]), tm = atoi(argv[6]);
    const char * xs = strchr(argv[6], 'x');
    const int tn = xs ? atoi(xs + 1) : 8;
    const size_t l0 = atoi(argv[7]), l1 = atoi(argv[8]);
    std::string opts = "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math";
    for (int i = 9; i < argc; i++) { opts += " "; opts += argv[i]; }
    const bool f16w = opts.find("-DF16W") != std::string::npos;
    const bool bbuf = opts.find("-DBBUF") != std::string::npos;  // the activations as a plain buffer, not an image
    const int Np = (N + tn - 1) / tn * tn;

    // random q8_0 weights (|w| ~ 0.05) and activations (|x| ~ 1)
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> ud(-1, 1);
    std::vector<int8_t> q((size_t) M * K);
    std::vector<uint16_t> d((size_t) M * (K / 32));
    for (auto & v : q) v = (int8_t) qd(rng);
    for (auto & v : d) v = to_half(0.0004f * (1.5f + ud(rng)));
    std::vector<uint16_t> xT((size_t) K * Np, 0);  // [k][n]
    std::vector<float> x((size_t) N * K);          // [n][k], the reference's
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) {
        const uint16_t h = to_half(ud(rng));
        xT[(size_t) k * Np + n] = h; x[(size_t) n * K + k] = from_half(h);
    }
    // noshuffle q: uint at (k/4) * M + row = the 4 int8 of row at k..k+3
    std::vector<uint32_t> qn((size_t) M * K / 4);
    std::vector<uint16_t> wh;
    for (int k4 = 0; k4 < K / 4; k4++) for (int r = 0; r < M; r++) {
        uint32_t u = 0;
        for (int j = 0; j < 4; j++) u |= (uint32_t) (uint8_t) q[(size_t) r * K + k4 * 4 + j] << (8 * j);
        qn[(size_t) k4 * M + r] = u;
    }
    std::vector<uint16_t> dn((size_t) M * (K / 32));  // half at (k/32) * M + row
    for (int b = 0; b < K / 32; b++) for (int r = 0; r < M; r++) dn[(size_t) b * M + r] = d[(size_t) r * (K / 32) + b];
    if (f16w) {  // half4 at (k/4) * M + row: w[row][k..k+3] * scale
        wh.resize((size_t) M * K);
        for (int k4 = 0; k4 < K / 4; k4++) for (int r = 0; r < M; r++) for (int j = 0; j < 4; j++) {
            const int k = k4 * 4 + j;
            wh[((size_t) k4 * M + r) * 4 + j] = to_half(q[(size_t) r * K + k] * from_half(d[(size_t) r * (K / 32) + k / 32]));
        }
    }

    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, nullptr));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CHECK(err);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue cq = clCreateCommandQueueWithProperties(ctx, dev, props, &err); CHECK(err);
    const char * s = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, nullptr, &err); CHECK(err);
    if (clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr) != CL_SUCCESS) {
        std::vector<char> log(1 << 16);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        fprintf(stderr, "build failed:\n%s\n", log.data());
        return 1;
    }
    cl_kernel kern = clCreateKernel(prog, kname, &err); CHECK(err);

    auto buf = [&](size_t bytes, const void * p) {
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (p ? CL_MEM_COPY_HOST_PTR : 0), bytes, (void *) p, &err); CHECK(err);
        return m;
    };
    cl_mem bq = f16w ? buf(wh.size() * 2, wh.data()) : buf(qn.size() * 4, qn.data());
    cl_mem bd = buf(dn.size() * 2, dn.data());
    cl_mem bx = buf(xT.size() * 2, xT.data());
    cl_mem bo = buf((size_t) Np * M * 4, nullptr);
    cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; desc.image_width = (size_t) K * Np / 4; desc.buffer = bx;
    cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err); CHECK(err);
    cl_ulong off = 0;
    int a = 0;
    CHECK(clSetKernelArg(kern, a++, sizeof(cl_mem), &bq));
    CHECK(clSetKernelArg(kern, a++, sizeof(cl_mem), &bd));
    CHECK(clSetKernelArg(kern, a++, sizeof(cl_mem), bbuf ? &bx : &img));
    CHECK(clSetKernelArg(kern, a++, sizeof(cl_mem), &bo));
    CHECK(clSetKernelArg(kern, a++, sizeof(int), &K));
    CHECK(clSetKernelArg(kern, a++, sizeof(int), &M));
    CHECK(clSetKernelArg(kern, a++, sizeof(int), &Np));
    CHECK(clSetKernelArg(kern, a++, sizeof(int), &N));
    CHECK(clSetKernelArg(kern, a++, sizeof(cl_ulong), &off));

    size_t gws[3] = { (size_t) Np / tn, (size_t) M / tm, 1 }, lws[3] = { l0, l1, 1 };
    double best = 1e30;
    for (int r = 0; r < 8; r++) {
        cl_event ev;
        CHECK(clEnqueueNDRangeKernel(cq, kern, 3, nullptr, gws, lws, 0, nullptr, &ev));
        CHECK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &t1, nullptr);
        clReleaseEvent(ev);
        if (r > 0 && (t1 - t0) * 1e-6 < best) best = (t1 - t0) * 1e-6;
    }
    std::vector<float> out((size_t) Np * M);
    CHECK(clEnqueueReadBuffer(cq, bo, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));

    // reference on a sample of outputs, in double
    double max_err = 0, max_ref = 0;
    std::uniform_int_distribution<int> rr(0, M - 1), rc(0, N - 1);
    for (int t = 0; t < 2000; t++) {
        const int r = rr(rng), c = rc(rng);
        double acc = 0;
        for (int k = 0; k < K; k++) acc += (double) q[(size_t) r * K + k] * from_half(d[(size_t) r * (K / 32) + k / 32]) * x[(size_t) c * K + k];
        max_err = std::fmax(max_err, std::fabs(acc - out[(size_t) c * M + r]));
        max_ref = std::fmax(max_ref, std::fabs(acc));
    }
    printf("%-28s M %5d K %5d N %4d: %7.3f ms  %.2f TFLOPS  max err %.2e (max |ref| %.2f)\n",
           kname, M, K, N, best, 2.0 * M * K * N / (best * 1e-3) * 1e-12, max_err, max_ref);
    return 0;
}
