#include <chrono>
#include <cassert>
#include <cstdio>
#include <functional>
#include <clocale>

#include <sycl/sycl.hpp>


#define STRINGIFY2(f) #f
#define STRINGIFY(f) STRINGIFY2(f)
#define SYCL_CHECK(f, ...) \
    try { \
        (f)(__VA_ARGS__); \
    } catch (sycl::exception const &e) { \
        fprintf(stderr, STRINGIFY(f) ": %s\n", e.what()); \
        std::abort(); \
    }


void copy1DWithKernel(sycl::queue &queue, const void *src, void *dst, size_t n_bytes) {
    assert(n_bytes % 4 == 0);
    const size_t n_ints = n_bytes / 4;
    const size_t local_range = 256;
    const size_t global_range = (n_ints + local_range - 1) / local_range * local_range;
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> it) {
            size_t i = it.get_global_id(0);
            if (i >= n_ints) return;
            ((int*) dst)[i] = ((const int*) src)[i];
        });
    });
}

void copy1DWithMemcpy(sycl::queue &queue, const void *src, void *dst, size_t n_bytes) {
    SYCL_CHECK(queue.memcpy, dst, src, n_bytes);
}

void copy2DWithKernel(sycl::queue &queue, const void *src, void *dst, size_t src_stride, size_t dst_stride, size_t chunk_size, size_t num_chunks) {
    assert(src_stride % 4 == 0);
    assert(dst_stride % 4 == 0);
    assert(chunk_size % 4 == 0);
    const auto src_stride_ints = src_stride / 4;
    const auto dst_stride_ints = dst_stride / 4;
    const auto chunk_size_ints = chunk_size / 4;
    const size_t local_range = 32;
    const size_t global_range = (num_chunks * chunk_size_ints + local_range - 1) / local_range * local_range;
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> it) {
            size_t i = it.get_global_id(0);
            const auto chunk_idx = i / chunk_size_ints;
            if (chunk_idx >= num_chunks) return;
            const auto offset = i % chunk_size_ints;
            ((int*) dst)[chunk_idx * dst_stride_ints + offset] = ((const int*) src)[chunk_idx * src_stride_ints + offset];
        });
    });
}

void copy2DWithMemcpy(sycl::queue &queue, const void *src, void *dst, size_t src_stride, size_t dst_stride, size_t chunk_size, size_t num_chunks) {
    if (src_stride == chunk_size && dst_stride == chunk_size) {
        SYCL_CHECK(queue.memcpy, dst, src, chunk_size * num_chunks);
    } else {
        for (size_t i = 0; i < num_chunks; ++i) {
            SYCL_CHECK(queue.memcpy, dst, src, chunk_size);
            src = (const char*) src + src_stride;
            dst = (char*) dst + dst_stride;
        }
    }
}

template<typename Fn, typename ...Params>
double runBenchmark(sycl::queue &queue, const Fn fn, const Params ...args) {
    const int n_warm = 2;
    const int n_time = 10;
    for (int i = 0; i < n_warm; ++i) { std::invoke(fn, queue, args...); }
    queue.wait();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_time; ++i) { std::invoke(fn, queue, args...); }
    queue.wait();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count() / n_time;
}

[[noreturn]] void usage() {
    fputs("Usage: sycl-nd-copy-bench 1D|2D <source> <dest>\n"
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

    const auto make_queue = [](int loc) {
        if (loc >= 0) {
            auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
            return sycl::queue(devices.at(loc), sycl::property::queue::in_order());
        } else {
            return sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
        }
    };
    auto source_queue = make_queue(source);
    auto dest_queue = make_queue(dest);
    auto copy_queue = make_queue(source >= 0 ? source : dest);

    printf("source is %s %s\n", source_queue.get_device().get_info<sycl::info::device::vendor>().c_str(), source_queue.get_device().get_info<sycl::info::device::name>().c_str());
    printf("dest is %s %s\n", dest_queue.get_device().get_info<sycl::info::device::vendor>().c_str(), dest_queue.get_device().get_info<sycl::info::device::name>().c_str());
    printf("\n");

    const size_t mem_size = 4ull << 30;

    void *mem_source;
    if (source >= 0) {
        mem_source = sycl::malloc_device(mem_size, source_queue);
    } else {
        mem_source = sycl::malloc_host(mem_size, source_queue);
    }

    void *mem_dest;
    if (dest >= 0) {
        mem_dest = sycl::malloc_device(mem_size, dest_queue);
    } else {
        mem_dest = sycl::malloc_host(mem_size, dest_queue);
    }

    if (dims == 1) {
        printf(" size                | sycl::memcpy      | kernel          \n");
        printf("---------------------+-----------------+-----------------\n");
        for (size_t n_bytes = 4; n_bytes <= mem_size; n_bytes *= 2) {
            const auto secondsMemcpy = runBenchmark(copy_queue, copy1DWithMemcpy, mem_source, mem_dest, n_bytes);
            const auto secondsKernel = runBenchmark(copy_queue, copy1DWithKernel, mem_source, mem_dest, n_bytes);
            printf(" %'13zu bytes | %'10.2f MB/s | %'10.2f MB/s\n", n_bytes, n_bytes * 1e-6 / secondsMemcpy, n_bytes * 1e-6 / secondsKernel);
        }
    } else if (dims == 2) {
        printf(" chunk size   | source stride      | dest stride        | #chunks  | volume       | sycl::memcpy      | kernel          \n");
        printf("==============+====================+====================+==========+==============+=================+=================\n");
        size_t chunk_size = 8;
        for (size_t chunk_size = 4; chunk_size <= 64ull << 10; chunk_size *= 2) {
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
                    const auto secondsMemcpy = runBenchmark(copy_queue, copy2DWithMemcpy, mem_source, mem_dest, sourceStride, destStride, chunk_size, num_chunks);
                    const auto secondsKernel = runBenchmark(copy_queue, copy2DWithKernel, mem_source, mem_dest, sourceStride, destStride, chunk_size, num_chunks);
                    printf(" %'6zu bytes | %'12zu bytes | %'12zu bytes | %'8zu | %'9zu KB | %'10.2f MB/s | %'10.2f MB/s\n", chunk_size, sourceStride, destStride, num_chunks, n_bytes / 1024, n_bytes * 1e-6 / secondsMemcpy, n_bytes * 1e-6 / secondsKernel);
                }
            }
            printf("==============+====================+====================+==========+==============+=================+=================\n");
        }
    }

    sycl::free(mem_source, source_queue);
    sycl::free(mem_dest, source_queue);
}

