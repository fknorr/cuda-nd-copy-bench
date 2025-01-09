cmake_minimum_required(VERSION 3.13)
project(nd_copy_bench CXX CUDA)

find_package(AdaptiveCpp REQUIRED)

add_executable(cuda_nd_copy_bench cuda-nd-copy-bench.cu)

add_executable(sycl_nd_copy_bench sycl-nd-copy-bench.cc)
target_compile_definitions(sycl_nd_copy_bench PRIVATE -DACPP_ALLOW_INSTANT_SUBMISSION=1)
add_sycl_to_target(TARGET sycl_nd_copy_bench SOURCES sycl-nd-copy-bench.cc)

