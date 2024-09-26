#include <chrono>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <functional>
#include <clocale>


#define STRINGIFY2(f) #f
#define STRINGIFY(f) STRINGIFY2(f)
#define CUDA_CHECK(f, ...) \
	if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) { \
		fprintf(stderr, STRINGIFY(f) ": %s\n", cudaGetErrorString(cuda_check_result)); \
        abort(); \
	}

__global__ void
copyKernel1D(const void *src, void *dst, size_t n_ints) {
    const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_ints) return;
    ((int*) dst)[i] = ((const int*) src)[i];
}

void copy1DWithKernel(const void *src, void *dst, size_t n_bytes) {
    assert(n_bytes % 4 == 0);
    const size_t n_ints = n_bytes / 4;
    const size_t threads_per_block = 256;
    const size_t num_blocks = (n_ints + threads_per_block - 1) / threads_per_block;
    copyKernel1D<<<num_blocks, threads_per_block>>>(src, dst, n_ints);
}

void copy1DWithMemcpy(const void *src, void *dst, size_t n_bytes) {
    CUDA_CHECK(cudaMemcpyAsync, dst, src, n_bytes, cudaMemcpyDefault);
}

__global__ void
copyKernel2D(const void *src, void *dst, size_t src_stride_ints, size_t dst_stride_ints, size_t chunk_size_ints, size_t num_chunks) {
    const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const auto chunk_idx = i / chunk_size_ints;
    if (chunk_idx >= num_chunks) return;
    const auto offset = i - (chunk_idx * chunk_size_ints);
    ((int*) dst)[chunk_idx * dst_stride_ints + offset] = ((const int*) src)[chunk_idx * src_stride_ints + offset];
}

void copy2DWithKernel(const void *src, void *dst, size_t src_stride, size_t dst_stride, size_t chunk_size, size_t num_chunks) {
    assert(src_stride % 4 == 0);
    assert(dst_stride % 4 == 0);
    assert(chunk_size % 4 == 0);
    const auto src_stride_ints = src_stride / 4;
    const auto dst_stride_ints = dst_stride / 4;
    const auto chunk_size_ints = chunk_size / 4;
    const size_t threads_per_block = 32;
    const size_t num_blocks = (num_chunks * chunk_size_ints + threads_per_block - 1) / threads_per_block;
    copyKernel2D<<<num_blocks, threads_per_block>>>(src, dst, src_stride_ints, dst_stride_ints, chunk_size_ints, num_chunks);
}

void copy2DWithMemcpy(const void *src, void *dst, size_t src_stride, size_t dst_stride, size_t chunk_size, size_t num_chunks) {
    CUDA_CHECK(cudaMemcpy2DAsync, dst, dst_stride, src, src_stride, chunk_size, num_chunks, cudaMemcpyDefault);
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

[[noreturn]] void usage() {
    fputs("Usage: cuda-nd-copy-bench 1D|2D <source> <dest>\n"
          "       where <source>/<dest> are H (host) or D0, D1, ... (devices)\n", stderr);
    exit(0);
}

int parse_loc(const char *arg) {
    if (strcmp(arg, "H") == 0) return -1;
    int device = -1;
    if (sscanf(arg, "D%d", &device) == 1) return device;
    usage();
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "en_US.UTF-8");

    if (argc != 4) usage();
    int dims = 0;
    if (sscanf(argv[1], "%dD", &dims) != 1 || dims < 1 || dims > 2) usage();
    int source = parse_loc(argv[2]);
    int dest = parse_loc(argv[3]);

    if (source >= 0 && dest >= 0 && source != dest) {
        CUDA_CHECK(cudaSetDevice, source);
        CUDA_CHECK(cudaDeviceEnablePeerAccess, dest, 0);
        CUDA_CHECK(cudaSetDevice, dest);
        CUDA_CHECK(cudaDeviceEnablePeerAccess, source, 0);
    }

    const size_t mem_size = 4ull << 30;

    if (dest >= 0) {
        // make sure that for "H -> D2" we setDevice(2) to allocate on host NUMA node close to D2
        CUDA_CHECK(cudaSetDevice, dest);
    }

    void *mem_source;
    if (source >= 0) {
        CUDA_CHECK(cudaSetDevice, source);
        CUDA_CHECK(cudaMalloc, &mem_source, mem_size);
    } else {
        CUDA_CHECK(cudaMallocHost, &mem_source, mem_size);
    }

    void *mem_dest;
    if (dest >= 0) {
        CUDA_CHECK(cudaSetDevice, dest);
        CUDA_CHECK(cudaMalloc, &mem_dest, mem_size);
    } else {
        CUDA_CHECK(cudaMallocHost, &mem_dest, mem_size);
    }

    if (dims == 1) {
        printf(" size                | cudaMemcpy      | kernel          \n");
        printf("---------------------+-----------------+-----------------\n");
        for (size_t n_bytes = 4; n_bytes <= mem_size; n_bytes *= 2) {
            const auto secondsMemcpy = runBenchmark(copy1DWithMemcpy, mem_source, mem_dest, n_bytes);
            const auto secondsKernel = runBenchmark(copy1DWithKernel, mem_source, mem_dest, n_bytes);
            printf(" %'13zu bytes | %'10.2f MB/s | %'10.2f MB/s\n", n_bytes, n_bytes * 1e-6 / secondsMemcpy, n_bytes * 1e-6 / secondsKernel);
        }
    } else if (dims == 2) {
        printf(" chunk size   | source stride      | dest stride        | #chunks  | volume       | cudaMemcpy      | kernel          \n");
        printf("==============+====================+====================+==========+==============+=================+=================\n");
        for (size_t chunk_size = 4; chunk_size <= 4096ull << 10; chunk_size *= 2) {
            for (int whichStride = 1; whichStride <= 3; ++whichStride) {
                if (whichStride != 1) {
                    printf("--------------+--------------------+--------------------+----------+--------------+-----------------+-----------------\n");
                }
                for (size_t stride = chunk_size; stride <= 4096 * chunk_size; stride *= 2) {
                    const auto sourceStride = (whichStride & 1) ? stride : chunk_size;
                    const auto destStride = (whichStride & 2) ? stride : chunk_size;
                    const size_t num_chunks = std::min<size_t>(64 << 10, mem_size / stride);
                    if (num_chunks < 2) continue;
                    const auto n_bytes = num_chunks * chunk_size;
                    const auto secondsMemcpy = runBenchmark(copy2DWithMemcpy, mem_source, mem_dest, sourceStride, destStride, chunk_size, num_chunks);
                    const auto secondsKernel = runBenchmark(copy2DWithKernel, mem_source, mem_dest, sourceStride, destStride, chunk_size, num_chunks);
                    printf(" %'6zu bytes | %'12zu bytes | %'12zu bytes | %'8zu | %'9zu KB | %'10.2f MB/s | %'10.2f MB/s\n", chunk_size, sourceStride, destStride, num_chunks, n_bytes / 1024, n_bytes * 1e-6 / secondsMemcpy, n_bytes * 1e-6 / secondsKernel);
                }
            }
            printf("==============+====================+====================+==========+==============+=================+=================\n");
        }
    }
}

