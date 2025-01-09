#include <chrono>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <functional>
#include <clocale>
#include <string>
#include <string_view>


#define STRINGIFY2(f) #f
#define STRINGIFY(f) STRINGIFY2(f)
#define CUDA_CHECK(f, ...) \
    if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) { \
        fprintf(stderr, STRINGIFY(f) ": %s\n", cudaGetErrorString(cuda_check_result)); \
        abort(); \
    }

template<typename Fragment, typename Index, bool SourceStrided, bool DestStrided>
__global__ void
copyKernel2D(const Fragment *src, Fragment *dst, Index src_stride, Index dst_stride, Index num_rows) {
    const Index chunk_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (chunk_idx >= num_rows) return;
    if (SourceStrided && DestStrided) {
        dst[chunk_idx * dst_stride] = src[chunk_idx * src_stride];
    } else if (SourceStrided) {
        dst[chunk_idx] = src[chunk_idx * src_stride];
    } else if (DestStrided) {
        dst[chunk_idx * dst_stride] = src[chunk_idx];
    } else {
        dst[chunk_idx] = src[chunk_idx];
    }
}

template<typename Fragment, typename Index, bool SourceStrided, bool DestStrided>
void copy2DWithKernel(const Fragment *src, Fragment *dst, Index src_stride, Index dst_stride, Index num_rows) {
    const Index threads_per_block = 32;
    const Index num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;
    copyKernel2D<Fragment, Index, SourceStrided, DestStrided><<<num_blocks, threads_per_block>>>(src, dst, src_stride, dst_stride, num_rows);
}

template<typename Fragment>
void copy2DWithMemcpy(const Fragment *src, Fragment *dst, size_t src_stride, size_t dst_stride, size_t num_rows) {
    CUDA_CHECK(cudaMemcpy2DAsync, dst, dst_stride * sizeof(Fragment), src, src_stride * sizeof(Fragment), sizeof(Fragment), num_rows, cudaMemcpyDefault);
}

template<typename ...Invocable>
double runBenchmark(Invocable &&...invocable) {
    const int n_warm = 2;
    const int n_time = 10;
    for (int i = 0; i < n_warm; ++i) { std::invoke(invocable...); }
    CUDA_CHECK(cudaDeviceSynchronize);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_time; ++i) { std::invoke(invocable...); }
    CUDA_CHECK(cudaDeviceSynchronize);
    const auto end = std::chrono::steady_clock::now();
    return (end - start) / std::chrono::duration<double>(n_time);
}

template<typename Fragment, typename Index, bool SourceStrided, bool DestStrided>
void with_fragment_and_strided_and_index(const Fragment *src, Fragment *dst, size_t src_stride, size_t dst_stride, size_t num_rows, const char *index_name) {
    auto t = runBenchmark(copy2DWithKernel<Fragment, Index, SourceStrided, DestStrided>, src, dst, src_stride, dst_stride, num_rows);
    auto n_bytes = sizeof(Fragment) * num_rows;
    auto mbs = n_bytes * 1e-6 / t;
    printf("kernel<%s> = %'10.2f MB/s\n", index_name, mbs);
}

template<typename Fragment, bool SourceStrided, bool DestStrided>
void with_fragment_and_strided(const Fragment *src, Fragment *dst, size_t src_stride, size_t dst_stride, size_t num_rows) {
    with_fragment_and_strided_and_index<Fragment, uint32_t, SourceStrided, DestStrided>(src, dst, src_stride, dst_stride, num_rows, "uint32_t");
    with_fragment_and_strided_and_index<Fragment, uint64_t, SourceStrided, DestStrided>(src, dst, src_stride, dst_stride, num_rows, "uint64_t");
}

template<typename Fragment>
void with_fragment(const Fragment *src, Fragment *dst, size_t src_stride, size_t dst_stride, size_t num_rows) {
    auto t = runBenchmark(copy2DWithMemcpy<Fragment>, src, dst, src_stride, dst_stride, num_rows);
    auto n_bytes = sizeof(Fragment) * num_rows;
    auto mbs = n_bytes * 1e-6 / t;
    printf("cudaMemcpy2D     = %'10.2f MB/s\n", mbs);

    if (src_stride > 1 && dst_stride > 1) {
        with_fragment_and_strided<Fragment, true, true>(src, dst, src_stride, dst_stride, num_rows);
    } else if (src_stride > 1) {
        with_fragment_and_strided<Fragment, true, false>(src, dst, src_stride, dst_stride, num_rows);
    } else if (dst_stride > 1) {
        with_fragment_and_strided<Fragment, false, true>(src, dst, src_stride, dst_stride, num_rows);
    } else {
        with_fragment_and_strided<Fragment, false, false>(src, dst, src_stride, dst_stride, num_rows);
    }
}

[[noreturn]] void usage() {
    fputs("Usage: cuda-single-bench <source> <dest> float|double <source-stride-elems> <dest-stride-elems> <rows>\n"
            "       where <source>/<dest> are H (host) or D0, D1, ... (devices)\n", stderr);
    exit(1);
}

int parse_loc(const char *arg) {
    if (strcmp(arg, "H") == 0) return -1;
    int device = -1;
    if (sscanf(arg, "D%d", &device) == 1) return device;
    usage();
}

int main(int argc, char **argv) {
    using namespace std::string_view_literals;

    setlocale(LC_ALL, "en_US.UTF-8");

    if (argc != 7) usage();
    int source = parse_loc(argv[1]);
    int dest = parse_loc(argv[2]);
    const auto fragment = argv[3];
    if (fragment != "float"sv && fragment != "double"sv) usage();
    size_t source_stride = std::stoul(argv[4]);
    size_t dest_stride = std::stoul(argv[5]);
    size_t num_rows = std::stoul(argv[6]);

    const size_t fragment_size = fragment == "float"sv ? sizeof(float) : sizeof(double);

    printf("2D copy with %'zu rows, datatype %s, %zu bytes\n", num_rows, fragment, fragment_size);
    printf("source: %2s, stride %'6zu fragments, %'6zu bytes\n", argv[1], source_stride, source_stride * fragment_size);
    printf("dest:   %2s, stride %'6zu fragments, %'6zu bytes\n", argv[2], dest_stride, dest_stride * fragment_size);
    printf("--------------------------------------------------\n");

    if (source >= 0 && dest >= 0 && source != dest) {
        CUDA_CHECK(cudaSetDevice, source);
        CUDA_CHECK(cudaDeviceEnablePeerAccess, dest, 0);
        CUDA_CHECK(cudaSetDevice, dest);
        CUDA_CHECK(cudaDeviceEnablePeerAccess, source, 0);
    }

    if (dest >= 0) {
        // make sure that for "H -> D2" we setDevice(2) to allocate on host NUMA node close to D2
        CUDA_CHECK(cudaSetDevice, dest);
    }

    const size_t source_mem_size = source_stride * num_rows * fragment_size;
    const size_t dest_mem_size = dest_stride * num_rows * fragment_size;

    void *mem_source;
    if (source >= 0) {
        CUDA_CHECK(cudaSetDevice, source);
        CUDA_CHECK(cudaMalloc, &mem_source, source_mem_size);
    } else {
        CUDA_CHECK(cudaMallocHost, &mem_source, source_mem_size);
    }

    void *mem_dest;
    if (dest >= 0) {
        CUDA_CHECK(cudaSetDevice, dest);
        CUDA_CHECK(cudaMalloc, &mem_dest, dest_mem_size);
    } else {
        CUDA_CHECK(cudaMallocHost, &mem_dest, dest_mem_size);
    }

    if (fragment == "float"sv) {
        with_fragment((const float*) mem_source, (float*) mem_dest, source_stride, dest_stride, num_rows);
    } else {
        with_fragment((const double*) mem_source, (double*) mem_dest, source_stride, dest_stride, num_rows);
    }
}

