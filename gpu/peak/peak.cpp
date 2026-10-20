// peak: the GPU's measured FMA throughput (half and float) and buffer read bandwidth, the roofline the prefill
// kernels are judged against.
//
//   peak [work items, default 1M]
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char * SRC = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// 8 independent chains of 8-wide FMAs per work item, ITERS times: 2 * 8 * 8 * ITERS flops each
#define BODY(T) \
    T y = (T)(a); \
    T x0 = (T)(get_local_id(0) * 1e-4f), x1 = x0 + y, x2 = x1 + y, x3 = x2 + y, \
      x4 = x3 + y, x5 = x4 + y, x6 = x5 + y, x7 = x6 + y; \
    for (int i = 0; i < ITERS; i++) { \
        x0 = mad(x0, y, y); x1 = mad(x1, y, y); x2 = mad(x2, y, y); x3 = mad(x3, y, y); \
        x4 = mad(x4, y, y); x5 = mad(x5, y, y); x6 = mad(x6, y, y); x7 = mad(x7, y, y); \
    } \
    T s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
kernel void fma_half(global half * out, float a) { BODY(half8) out[get_global_id(0)] = s.s0 + s.s7; }
kernel void fma_float(global float * out, float a) { BODY(float8) out[get_global_id(0)] = s.s0 + s.s7; }
kernel void read_f4(global const float4 * in, global float * out, int n4) {
    float4 acc = 0;
    for (int i = get_global_id(0); i < n4; i += get_global_size(0)) acc += in[i];
    out[get_global_id(0)] = acc.x + acc.y + acc.z + acc.w;
}
)CL";

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s: %d\n", #x, e_); exit(1); } } while (0)

static double run(cl_command_queue q, cl_kernel k, size_t global, size_t local, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        cl_event ev;
        CHECK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, &ev));
        CHECK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1;
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr));
        clReleaseEvent(ev);
        if (r > 0 && (t1 - t0) * 1e-9 < best) best = (t1 - t0) * 1e-9;  // the first run warms up
    }
    return best;
}

int main(int argc, char ** argv) {
    const size_t global = argc > 1 ? strtoul(argv[1], nullptr, 10) : (1 << 20);
    const int iters = 256;
    cl_platform_id plat; cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &plat, nullptr));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    char name[256]; cl_uint cus, mhz;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, nullptr);
    printf("%s, %u compute units, %u MHz\n", name, cus, mhz);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CHECK(err);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &err); CHECK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, nullptr, &err); CHECK(err);
    std::string opts = "-cl-fast-relaxed-math -DITERS=" + std::to_string(iters);
    if (clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr) != CL_SUCCESS) {
        std::vector<char> log(1 << 16);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        fprintf(stderr, "%s\n", log.data());
        return 1;
    }
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, global * sizeof(float), nullptr, &err); CHECK(err);
    const float a = 0.999f;
    for (const char * kn : { "fma_half", "fma_float" }) {
        cl_kernel k = clCreateKernel(prog, kn, &err); CHECK(err);
        CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &out));
        CHECK(clSetKernelArg(k, 1, sizeof(float), &a));
        for (size_t local : { 64, 128, 256 }) {
            const double s = run(q, k, global, local, 4);
            printf("%-9s local %3zu: %.2f TFLOPS\n", kn, local, 2.0 * 8 * 8 * iters * global / s * 1e-12);
        }
        clReleaseKernel(k);
    }
    const size_t bytes = 512u << 20;
    cl_mem in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err); CHECK(err);
    std::vector<float> zero(bytes / sizeof(float), 1.0f);
    CHECK(clEnqueueWriteBuffer(q, in, CL_TRUE, 0, bytes, zero.data(), 0, nullptr, nullptr));
    cl_kernel k = clCreateKernel(prog, "read_f4", &err); CHECK(err);
    const int n4 = (int) (bytes / 16);
    const size_t g = 1 << 18;
    CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &in));
    CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &out));
    CHECK(clSetKernelArg(k, 2, sizeof(int), &n4));
    const double s = run(q, k, g, 128, 4);
    printf("read      : %.1f GB/s\n", bytes / s * 1e-9);
    return 0;
}
